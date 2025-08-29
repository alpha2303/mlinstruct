from pathlib import Path
from typing import Iterable, Optional

from ...utils.funcs import is_dependency_installed

if not is_dependency_installed("torch"):
    raise ImportError("Torch is not available")

from ..model_proxy.base_model_proxy import BaseModelProxy
from ..data.enum import ModelFormat
from ...utils.exception import ModelProxyError

import torchinfo
import torch
from torch.utils.data import DataLoader


class TorchModelProxy(BaseModelProxy):
    """PyTorch model proxy for handling model training and evaluation.

    This class provides an interface for training and evaluating PyTorch models
    while abstracting away the details of the underlying framework.

    Inherits from BaseModelProxy.

    Args:
        model (torch.nn.Module): The PyTorch model to be proxied.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        loss_fn (torch.nn.modules.loss._Loss): The loss function for training the model.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The learning rate scheduler for the model. Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        self.__model = model
        self.__optimizer = optimizer
        self.__loss_fn = loss_fn
        self.__scheduler = scheduler
        super().__init__()

    def load_model(self, model_file_path: Path) -> None:
        """Load model from a saved model checkpoint file.

        Args:
            model_file_path (Path): The path to the model file.
        """
        extension = model_file_path.suffix.lower()
        match extension:
            case ModelFormat.PT.value:
                checkpoint = torch.load(model_file_path)
                if "model_state_dict" not in checkpoint:
                    raise ModelProxyError(
                        "Model state dict is missing from checkpoint."
                    )
                self.__model.load_state_dict(checkpoint.get("model_state_dict"))

                if "optimizer_state_dict" in checkpoint:
                    self.__optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )
            # TODO: Add support for ONNX model format
            case _:
                raise ModelProxyError(
                    f"Unsupported model file format: {extension}."
                )

        self.__model.eval()

    def save_model(
        self,
        epoch: int,
        save_dir_path: Path,
        model_name: str,
        loss: float,
        save_format: ModelFormat,
        **kwargs,
    ) -> None:
        """Save model to a file.

        Args:
            epoch (int): The current epoch number.
            save_path (Path): The folder path to save the model.
            model_name (str): The name of the model.
            loss (float): The current loss value.
            save_format (ModelFormat): The format to save the model. Defaults to NATIVE.

        Raises:
            ModelProxyError:
                - If the save directory path does not exist.
                - If the model save format is unsupported.
        """
        if not save_dir_path.exists():
            raise ModelProxyError(
                "Model save path does not exist. If you are running the save method directly, ensure that the save path is valid."
            )
        
        match save_format:
            case ModelFormat.PT:
                model_object = {
                    "epoch": epoch,
                    "model_state_dict": self.__model.state_dict(),
                    "optimizer_state_dict": self.__optimizer.state_dict(),
                    "loss": loss,
                }

                torch.save(model_object, save_dir_path.joinpath(f"{model_name}.pt"))
            case ModelFormat.ONNX:
                torch.onnx.export(
                    model=self.__model,
                    f=save_dir_path.joinpath(f"{model_name}.onnx"),
                    dynamo=True,
                )
            case _:
                raise ModelProxyError(
                    f"Unsupported model save format: {save_format.value}."
                )

    def get_lr(self) -> float:
        """Get the current learning rate from PyTorch optimizer used by the model.

        Returns:
            float: The current learning rate.
        """
        return self.__optimizer.param_groups[0]["lr"]

    def has_scheduler(self) -> bool:
        """Check if the model has a learning rate scheduler set up.

        Returns:
            bool: True if the model has a scheduler, False otherwise.
        """
        return isinstance(self.__scheduler, torch.optim.lr_scheduler.LRScheduler)

    def scheduler_step(self, avg_vloss: float) -> None:
        """Perform a step of the learning rate scheduler if it exists.

        Args:
            avg_vloss (float): The average validation loss for the current epoch.
        """
        if not self.has_scheduler():
            raise ModelProxyError(
                "Model Proxy does not have a valid scheduler configured."
            )

        if isinstance(self.__scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.__scheduler.step(avg_vloss)
        else:
            self.__scheduler.step()  # type: ignore

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
        try:
            self.__model.train()
            for _, data in enumerate(train_data):
                X_batch, Y_batch = data
                Y_pred = self.__model(X_batch)
                loss = self.__loss_fn(Y_pred, Y_batch)
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()

                running_loss += loss.item()
        except Exception as e:
            raise e

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

        try:
            self.__model.eval()
            with torch.no_grad():
                for _, vdata in enumerate(test_data):
                    vX_batch, vY_batch = vdata
                    vY_pred: torch.Tensor = self.__model(vX_batch)
                    vloss = self.__loss_fn(vY_pred, vY_batch)
                    running_vloss += vloss.item()
        except Exception as e:
            raise e

        return running_vloss / len(test_data)

    def summary(self) -> torchinfo.ModelStatistics:
        """Get a summary of the model architecture.

        Returns:
            torchinfo.ModelStatistics: A summary of the model.
        """
        return torchinfo.summary(self.__model)
