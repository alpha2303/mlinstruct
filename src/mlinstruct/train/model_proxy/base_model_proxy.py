from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class BaseModelProxy(ABC):
    """Base class for model proxies.

    This class defines the interface for model proxies, which are responsible for
    interacting with the underlying machine learning models.

    Model proxies are intended to act as adapters between models from
    different frameworks and the trainer logic.
    """

    @abstractmethod
    def load_weights(self, model_file_path: Path) -> None:
        """Load model weights from a saved model checkpoint file.

        Args:
            model_file_path (Path): The path to the model file.
        """
        pass

    @abstractmethod
    def save_weights(
        self, epoch: int, save_dir_path: Path, model_name: str, *args, **kwargs
    ) -> None:
        """
        Save model weights to a file.

        Args:
            epoch (int): The current epoch number.
            save_folder_path (Path): The folder path to save the model weights.
            model_name (str): The name of the model.
        """
        pass

    @abstractmethod
    def get_lr(self) -> float:
        """
        Get the current learning rate.

        Returns:
            float: The current learning rate.
        """
        pass

    @abstractmethod
    def has_scheduler(self) -> bool:
        """
        Check if the model has a learning rate scheduler.

        Returns:
            bool: True if the model has a scheduler, False otherwise.
        """
        pass

    @abstractmethod
    def scheduler_step(self, avg_vloss: float) -> None:
        """
        Update the learning rate scheduler.

        Args:
            avg_vloss (float): The average validation loss.
        """
        pass

    @abstractmethod
    def train_one_epoch(self, train_data: Iterable) -> float:
        """
        Train the model for one epoch.

        Args:
            train_data (Iterable): The training data.

        Returns:
            float: The average training loss for the epoch.
        """
        pass

    @abstractmethod
    def validate(self, test_data: Iterable) -> float:
        """
        Validate the model on the given test data.

        Args:
            test_data (Iterable): The test data.

        Returns:
            float: The average validation loss for the test data.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        pass
