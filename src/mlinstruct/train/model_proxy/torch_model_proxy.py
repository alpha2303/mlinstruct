from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

from ..model_proxy.base_model_proxy import BaseModelProxy
from ..utils.device import move_to_device, resolve_device
from ...utils.exception import ModelProxyError

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    import torchinfo


class TorchModelProxy(BaseModelProxy):
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
        scheduler (Optional[LRScheduler], optional): The learning rate scheduler for the model. Defaults to None.
        model_name (Optional[str], optional): The name of the model. Defaults to the model class name.
        device (Optional[Union[str, torch.device]], optional): The device to train on. Defaults to the
            current accelerator if one is available, else CPU.

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: Optional[LRScheduler] = None,
        model_name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self._device = resolve_device(device)
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._scheduler = scheduler
        self._model_name = model_name or type(model).__name__
        super().__init__()

    @property
    def device(self) -> torch.device:
        """The device the model is trained and evaluated on."""
        return self._device

    def load_weights(self, model_file_path: Path) -> None:
        """Load model weights from a saved model checkpoint file.

        Args:
            model_file_path (Path): The path to the model file.
        """
        checkpoint = torch.load(model_file_path)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._model.eval()

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
                "Model save path does not exist. If you are running the save method directly, ensure that the save path is valid."
            )

        model_object = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "loss": loss,
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
            raise ModelProxyError(
                "Model Proxy does not have a valid scheduler configured."
            )

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
            Y_pred = self._model(X_batch)
            loss = self._loss_fn(Y_pred, Y_batch)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

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
            raise ModelProxyError(
                "Test Input is not an instance of torch.utils.data.DataLoader"
            )

        running_vloss: float = 0.0

        self._model.eval()
        with torch.no_grad():
            for batch in test_data:
                vX_batch, vY_batch = move_to_device(batch, self._device)
                vY_pred: torch.Tensor = self._model(vX_batch)
                vloss = self._loss_fn(vY_pred, vY_batch)
                running_vloss += vloss.item()

        return running_vloss / len(test_data)

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
