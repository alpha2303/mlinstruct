from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mlinstruct import __version__
from mlinstruct.train.model_proxy.onnx_exportable import OnnxExportable
from mlinstruct.train.utils.device import move_to_device, resolve_device
from mlinstruct.utils.exception import ModelProxyError
from mlinstruct.utils.optional_deps import require

GeneratorLossFn = Callable[[torch.Tensor], torch.Tensor]
DiscriminatorLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def non_saturating_generator_loss(d_pred_fake: torch.Tensor) -> torch.Tensor:
    """-log(D(G(z))); the standard non-saturating GAN generator loss."""
    return nn.functional.binary_cross_entropy(d_pred_fake, torch.ones_like(d_pred_fake))


def bce_discriminator_loss(d_pred_real: torch.Tensor, d_pred_fake: torch.Tensor) -> torch.Tensor:
    """BCE loss for a discriminator: real -> 1, fake -> 0."""
    real_loss = nn.functional.binary_cross_entropy(d_pred_real, torch.ones_like(d_pred_real))
    fake_loss = nn.functional.binary_cross_entropy(d_pred_fake, torch.zeros_like(d_pred_fake))
    return real_loss + fake_loss


class GANModelProxy(OnnxExportable):
    """PyTorch model proxy for a vanilla (unconditional, single-G/single-D) GAN.

    Does not implement BaseModelProxy: a GAN has two networks, two optimizers,
    and no single scalar validation loss with a meaningful "better" direction,
    so it is a new, independent class rather than a reshaping of
    BaseModelProxy's single-model contract.

    Args:
        generator (nn.Module): The generator network, mapping latent noise to samples.
        discriminator (nn.Module): The discriminator network, mapping samples to a
            real/fake prediction.
        generator_optimizer (torch.optim.Optimizer): Optimizer for the generator.
        discriminator_optimizer (torch.optim.Optimizer): Optimizer for the discriminator.
        latent_dim (int): Dimensionality of the generator's noise input.
        generator_loss_fn (GeneratorLossFn): Maps the discriminator's prediction on a
            fake batch to a scalar generator loss. Defaults to the non-saturating loss.
        discriminator_loss_fn (DiscriminatorLossFn): Maps the discriminator's
            predictions on a real and a fake batch to a scalar discriminator loss.
            Defaults to BCE.
        model_name (Optional[str]): The name of the model. Defaults to the generator
            class name.
        device (Optional[Union[str, torch.device]]): The device to train on. Defaults
            to the current accelerator if one is available, else CPU.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        latent_dim: int,
        generator_loss_fn: GeneratorLossFn = non_saturating_generator_loss,
        discriminator_loss_fn: DiscriminatorLossFn = bce_discriminator_loss,
        model_name: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self._device = resolve_device(device)
        self._generator = generator.to(self._device)
        self._discriminator = discriminator.to(self._device)
        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        self._latent_dim = latent_dim
        self._generator_loss_fn = generator_loss_fn
        self._discriminator_loss_fn = discriminator_loss_fn
        self._model_name = model_name or type(generator).__name__
        super().__init__()

    @property
    def device(self) -> torch.device:
        """The device the models are trained and evaluated on."""
        return self._device

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """Sample a batch of latent noise vectors.

        Args:
            batch_size (int): Number of noise vectors to sample.

        Returns:
            torch.Tensor: A (batch_size, latent_dim) tensor on the proxy's device.
        """
        return torch.randn(batch_size, self._latent_dim, device=self._device)

    def train_one_batch(self, real_batch: torch.Tensor, n_critic: int = 1) -> tuple[float, float]:
        """Run n_critic discriminator steps, then one generator step, on real_batch.

        Fake batches used for the discriminator step are detached so discriminator
        gradients don't flow into the generator.

        Args:
            real_batch (torch.Tensor): A batch of real samples.
            n_critic (int): Number of discriminator steps to run per generator step.

        Returns:
            tuple[float, float]: (generator_loss, mean_discriminator_loss_over_n_critic_steps).
        """
        real_batch = move_to_device(real_batch, self._device)
        self._generator.train()
        self._discriminator.train()

        batch_size = real_batch.size(0)

        d_losses = []
        for _ in range(n_critic):
            self._discriminator_optimizer.zero_grad()
            fake_batch = self._generator(self.sample_noise(batch_size)).detach()
            d_pred_real = self._discriminator(real_batch)
            d_pred_fake = self._discriminator(fake_batch)
            d_loss = self._discriminator_loss_fn(d_pred_real, d_pred_fake)
            d_loss.backward()
            self._discriminator_optimizer.step()
            d_losses.append(d_loss.item())

        self._generator_optimizer.zero_grad()
        fake_batch = self._generator(self.sample_noise(batch_size))
        d_pred_fake_for_generator = self._discriminator(fake_batch)
        g_loss = self._generator_loss_fn(d_pred_fake_for_generator)
        g_loss.backward()
        self._generator_optimizer.step()

        return g_loss.item(), sum(d_losses) / len(d_losses)

    def generate(self, n_samples: int) -> torch.Tensor:
        """Generator forward pass in eval mode, under no_grad.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: The generated samples.
        """
        self._generator.eval()
        with torch.no_grad():
            return self._generator(self.sample_noise(n_samples))

    def save_weights(
        self,
        epoch: int,
        save_dir_path: Path,
        model_name: str,
        *,
        g_loss: float,
        d_loss: float,
        **kwargs: Any,
    ) -> Path:
        """Write generator+discriminator state_dicts and optimizer states to one file.

        Args:
            epoch (int): The current epoch number.
            save_dir_path (Path): The folder path to save the checkpoint in.
            model_name (str): The stem of the checkpoint filename, without extension.
            g_loss (float): The current generator loss value.
            d_loss (float): The current discriminator loss value.

        Returns:
            Path: The path the checkpoint was written to.
        """
        if not save_dir_path.exists():
            raise ModelProxyError(
                "Model save path does not exist. If you are running the save method "
                "directly, ensure that the save path is valid."
            )

        model_object = {
            "epoch": epoch,
            "generator_state_dict": self._generator.state_dict(),
            "discriminator_state_dict": self._discriminator.state_dict(),
            "generator_optimizer_state_dict": self._generator_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": self._discriminator_optimizer.state_dict(),
            "g_loss": g_loss,
            "d_loss": d_loss,
            "mlinstruct_version": __version__,
        }

        model_path: Path = save_dir_path.joinpath(f"{model_name}.pt")
        torch.save(model_object, model_path)
        return model_path

    def load_checkpoint(self, model_file_path: Path) -> int:
        """Load generator, discriminator, and optimizer state from a checkpoint file.

        Args:
            model_file_path (Path): The path to the checkpoint file.

        Returns:
            int: The epoch recorded in the checkpoint.
        """
        checkpoint = torch.load(model_file_path, map_location=self._device, weights_only=True)
        self._generator.load_state_dict(checkpoint["generator_state_dict"])
        self._discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self._generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        self._discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state_dict"]
        )
        return checkpoint["epoch"]

    def get_model_name(self) -> str:
        """Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self._model_name

    def export_onnx(
        self, save_path: Path, input_sample: Any, *, dynamo: bool = True, **kwargs: Any
    ) -> Path:
        """Export the generator (only) to ONNX.

        Args:
            save_path (Path): The file path to write the ONNX model to.
            input_sample (Any): A representative noise batch, e.g.
                gan_proxy.sample_noise(n).
            dynamo (bool, optional): Use torch's dynamo-based exporter. Defaults to True.
            **kwargs (Any): Passed through to torch.onnx.export.

        Returns:
            Path: The path the ONNX model was written to.

        Raises:
            ModelProxyError: If save_path's parent directory does not exist.
        """
        require("onnx", extra="onnx", symbol="GANModelProxy.export_onnx")
        if dynamo:
            require("onnxscript", extra="onnx", symbol="GANModelProxy.export_onnx(dynamo=True)")
        if not save_path.parent.exists():
            raise ModelProxyError("ONNX export path's parent directory does not exist.")

        was_training = self._generator.training
        self._generator.eval()
        try:
            sample = move_to_device(input_sample, self._device)
            torch.onnx.export(self._generator, sample, str(save_path), dynamo=dynamo, **kwargs)
        finally:
            self._generator.train(was_training)

        return save_path
