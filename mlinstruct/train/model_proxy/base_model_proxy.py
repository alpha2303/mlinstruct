from pathlib import Path
from typing import Iterable


class BaseModelProxy:
    """Base class for model proxies.

    This class defines the interface for model proxies, which are responsible for
    interacting with the underlying machine learning models.

    Model proxies are intended to act as adapters between models from
    different frameworks and the trainer logic.
    """

    def load_weights(self, model_file_path: Path) -> None:
        """Load model weights from a saved model checkpoint file.

        Args:
            model_file_path (Path): The path to the model file.
        """
        raise NotImplementedError()

    def save_weights(
        self, epoch: int, loss: float, save_folder_path: Path, model_name: str
    ) -> None:
        """
        Save model weights to a file.

        Args:
            epoch (int): The current epoch number.
            loss (float): The current loss value.
            save_folder_path (Path): The folder path to save the model weights.
            model_name (str): The name of the model.
        """
        raise NotImplementedError()

    def get_lr(self) -> float:
        """
        Get the current learning rate.

        Returns:
            float: The current learning rate.
        """
        raise NotImplementedError()

    def has_scheduler(self) -> bool:
        """
        Check if the model has a learning rate scheduler.

        Returns:
            bool: True if the model has a scheduler, False otherwise.
        """
        raise NotImplementedError()

    def step(self, avg_vloss: float) -> None:
        """
        Update the learning rate scheduler.

        Args:
            avg_vloss (float): The average validation loss.
        """
        raise NotImplementedError()

    def train_one_epoch(self, train_data: Iterable) -> float:
        """
        Train the model for one epoch.

        Args:
            train_data (Iterable): The training data.

        Returns:
            float: The average training loss for the epoch.
        """
        raise NotImplementedError()

    def validate(self, test_data: Iterable) -> float:
        """
        Validate the model on the given test data.

        Args:
            test_data (Iterable): The test data.

        Returns:
            float: The average validation loss for the test data.
        """
        raise NotImplementedError()

    def summary(self) -> None:
        """
        Print a summary of the model architecture and parameters.
        """
        raise NotImplementedError()
