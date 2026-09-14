import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from mlinstruct import __version__
from mlinstruct.train.model_proxy.base_model_proxy import BaseModelProxy
from mlinstruct.train.model_proxy.onnx_exportable import OnnxExportable
from mlinstruct.train.utils.device import move_to_device, resolve_device
from mlinstruct.utils.exception import ModelProxyError
from mlinstruct.utils.optional_deps import require

if TYPE_CHECKING:
    import torchinfo


class TorchModelProxy(BaseModelProxy, OnnxExportable):
    """PyTorch model proxy for handling model training and evaluation.

    This class provides an interface for training and evaluating PyTorch models
    while abstracting away the details of the underlying framework.

    Inherits from BaseModelProxy.

    The proxy expects each batch yielded by a DataLoader to be a 2-tuple of
    ``(inputs, targets)``; both elements are moved to the proxy's device
    before the forward pass.

    Args:
        model (nn.Module): The PyTorch model to be proxied.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        loss_fn (nn.Module): The loss function for training the model.
        scheduler (Optional[LRScheduler], optional): The learning rate scheduler for the
            model. Defaults to None.
        model_name (Optional[str], optional): The name of the model. Defaults to the
            model class name.
        device (Optional[Union[str, torch.device]], optional): The device to train on.
            Defaults to the current accelerator if one is available, else CPU.
        use_amp (bool, optional): Whether to train with automatic mixed precision.
            Defaults to False.
        amp_dtype (Optional[torch.dtype], optional): The autocast dtype to use when
            use_amp is True. Defaults to bfloat16 if the device is CUDA and supports
            it, else float16 on CUDA or bfloat16 on CPU.
        max_grad_norm (Optional[float], optional): If set, gradients are clipped to
            this max norm before each optimizer step. Defaults to None (no clipping).

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: LRScheduler | None = None,
        model_name: str | None = None,
        device: str | torch.device | None = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        self._device = resolve_device(device)
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._scheduler = scheduler
        self._model_name = model_name or type(model).__name__
        self._use_amp = use_amp
        self._amp_dtype = amp_dtype or self._default_amp_dtype()
        self._max_grad_norm = max_grad_norm
        self._scaler = torch.amp.GradScaler(
            self._device.type, enabled=use_amp and self._amp_dtype == torch.float16
        )
        super().__init__()

    def _default_amp_dtype(self) -> torch.dtype:
        if self._device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if self._device.type == "cuda":
            return torch.float16
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        """The device the model is trained and evaluated on."""
        return self._device

    def load_checkpoint(self, model_file_path: Path) -> int:
        """Load model, optimizer, and (if present) scheduler state from a checkpoint file.

        Does not change the model's train/eval mode.

        Args:
            model_file_path (Path): The path to the checkpoint file.

        Returns:
            int: The epoch recorded in the checkpoint.
        """
        checkpoint = torch.load(model_file_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler_state_dict is not None and self.has_scheduler():
            self._scheduler.load_state_dict(scheduler_state_dict)  # type: ignore

        scaler_state_dict = checkpoint.get("scaler_state_dict")
        if scaler_state_dict is not None:
            self._scaler.load_state_dict(scaler_state_dict)

        return checkpoint["epoch"]

    def load_weights(self, model_file_path: Path) -> None:
        """Deprecated alias for load_checkpoint.

        Args:
            model_file_path (Path): The path to the checkpoint file.
        """
        warnings.warn(
            "TorchModelProxy.load_weights is deprecated; use load_checkpoint instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.load_checkpoint(model_file_path)

    def save_weights(
        self, epoch: int, save_dir_path: Path, model_name: str, loss: float, **kwargs
    ) -> Path:
        """Save model weights to a file.

        Args:
            epoch (int): The current epoch number.
            save_path (Path): The folder path to save the model weights.
            model_name (str): The stem of the checkpoint filename, without extension.
            loss (float): The current loss value.

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
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": (
                self._scheduler.state_dict() if self.has_scheduler() else None  # type: ignore
            ),
            "scaler_state_dict": self._scaler.state_dict(),
            "loss": loss,
            "mlinstruct_version": __version__,
        }

        model_path: Path = save_dir_path.joinpath(f"{model_name}.pt")
        torch.save(model_object, model_path)
        return model_path

    def get_lr(self) -> float:
        """Get the current learning rate from PyTorch optimizer used by the model.

        Returns:
            float: The current learning rate.
        """
        return self._optimizer.param_groups[0]["lr"]

    def has_scheduler(self) -> bool:
        """Check if the model has a learning rate scheduler set up.

        Returns:
            bool: True if the model has a scheduler, False otherwise.
        """
        return isinstance(self._scheduler, LRScheduler)

    def scheduler_step(self, avg_vloss: float) -> None:
        """Perform a step of the learning rate scheduler if it exists.

        Args:
            avg_vloss (float): The average validation loss for the current epoch.
        """
        if not self.has_scheduler():
            raise ModelProxyError("Model Proxy does not have a valid scheduler configured.")

        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self._scheduler.step(avg_vloss)
        else:
            self._scheduler.step()  # type: ignore

    def train_one_epoch(self, train_data: Iterable) -> float:
        """Train the model for one epoch.

        Args:
            train_data (Iterable): The training data loader.

        Returns:
            float: The average training loss for the epoch.

        Raises:
            ModelProxyError: If the training data is not a DataLoader instance.
        """
        if not isinstance(train_data, DataLoader):
            raise ModelProxyError(
                "Training Input is not an instance of torch.utils.data.DataLoader"
            )

        running_loss = 0.0
        self._model.train()
        for batch in train_data:
            X_batch, Y_batch = move_to_device(batch, self._device)
            self._optimizer.zero_grad()

            with torch.autocast(
                device_type=self._device.type, dtype=self._amp_dtype, enabled=self._use_amp
            ):
                Y_pred = self._model(X_batch)
                loss = self._loss_fn(Y_pred, Y_batch)

            self._scaler.scale(loss).backward()

            if self._max_grad_norm is not None:
                self._scaler.unscale_(self._optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)

            self._scaler.step(self._optimizer)
            self._scaler.update()

            running_loss += loss.item()

        return running_loss / len(train_data)

    def validate(self, test_data: Iterable) -> float:
        """Validate the model on the test dataset.

        Args:
            test_data (Iterable): The test data loader.

        Returns:
            float: The average validation loss for the epoch.

        Raises:
            ModelProxyError: If the test data is not a DataLoader instance.
        """
        if not isinstance(test_data, DataLoader):
            raise ModelProxyError("Test Input is not an instance of torch.utils.data.DataLoader")

        running_vloss: float = 0.0

        self._model.eval()
        with torch.no_grad():
            for batch in test_data:
                vX_batch, vY_batch = move_to_device(batch, self._device)
                vY_pred: torch.Tensor = self._model(vX_batch)
                vloss = self._loss_fn(vY_pred, vY_batch)
                running_vloss += vloss.item()

        return running_vloss / len(test_data)

    def predict(self, data: Iterable) -> tuple[np.ndarray, np.ndarray]:
        """Run inference over a DataLoader and collect true/predicted values.

        Args:
            data (Iterable): The data loader to predict over.

        Returns:
            tuple[np.ndarray, np.ndarray]: The concatenated (y_true, y_pred) arrays,
                moved to CPU.

        Raises:
            ModelProxyError: If data is not a DataLoader instance.
        """
        if not isinstance(data, DataLoader):
            raise ModelProxyError("Predict Input is not an instance of torch.utils.data.DataLoader")

        y_true_batches = []
        y_pred_batches = []

        self._model.eval()
        with torch.inference_mode():
            for batch in data:
                X_batch, Y_batch = move_to_device(batch, self._device)
                Y_pred = self._model(X_batch)
                y_true_batches.append(Y_batch.cpu().numpy())
                y_pred_batches.append(Y_pred.cpu().numpy())

        return np.concatenate(y_true_batches), np.concatenate(y_pred_batches)

    def get_model_name(self) -> str:
        """Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self._model_name

    def summary(self) -> "torchinfo.ModelStatistics":
        """Get a summary of the model architecture.

        Returns:
            torchinfo.ModelStatistics: A summary of the model.
        """
        import torchinfo

        return torchinfo.summary(self._model)

    def export_onnx(
        self, save_path: Path, input_sample: Any, *, dynamo: bool = True, **kwargs: Any
    ) -> Path:
        """Export the model to ONNX.

        Args:
            save_path (Path): The file path to write the ONNX model to.
            input_sample (Any): A representative single batch used for tracing/shape
                inference, in whatever type the model's forward pass expects.
            dynamo (bool, optional): Use torch's dynamo-based exporter. Defaults to True.
            **kwargs (Any): Passed through to torch.onnx.export (e.g. input_names,
                dynamic_shapes, opset_version).

        Returns:
            Path: The path the ONNX model was written to.

        Raises:
            ModelProxyError: If save_path's parent directory does not exist.
        """
        require("onnx", extra="onnx", symbol="TorchModelProxy.export_onnx")
        if dynamo:
            require("onnxscript", extra="onnx", symbol="TorchModelProxy.export_onnx(dynamo=True)")
        if not save_path.parent.exists():
            raise ModelProxyError("ONNX export path's parent directory does not exist.")

        was_training = self._model.training
        self._model.eval()
        try:
            sample = move_to_device(input_sample, self._device)
            torch.onnx.export(self._model, sample, str(save_path), dynamo=dynamo, **kwargs)
        finally:
            self._model.train(was_training)

        return save_path
