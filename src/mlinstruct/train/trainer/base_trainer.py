from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Self

from ..data_payload.base_data_payload import BaseDataPayload
from ..data.train_result import TrainResult
from ..data.enum import ModelFormat
from ..model_proxy.base_model_proxy import BaseModelProxy
from ..utils.early_stopper import EarlyStopper
from ..utils.checkpoint_writer import CheckpointWriter

DEFAULT_SAVE_PATH: Path = Path("./Models")
_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M"


class BaseTrainer(ABC):
    """Base class for all trainers."""

    __model_proxy: BaseModelProxy
    __data_payload: BaseDataPayload
    __checkpoint_writer: CheckpointWriter
    __root_save_dir_path: Path
    __early_stopper: Optional[EarlyStopper]
    __save_format: ModelFormat

    @abstractmethod
    def train(self, max_epochs: int) -> TrainResult:
        """Train the model.

        Args:
            max_epochs (int): The maximum number of training epochs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The training and validation losses.
        """
        pass

    def regenerate_model_save_path(self) -> None:
        """Regenerate the model save path based on the current timestamp."""
        self.__model_save_dir_path = self.__root_save_dir_path.joinpath(
            datetime.now().strftime(_TIMESTAMP_FORMAT)
        )
        if not self.__model_save_dir_path.exists():
            self.__model_save_dir_path.mkdir(parents=True)

    def add_early_stop(self, patience: int = 2, min_delta: float = 0.01) -> Self:
        """Add early stopping to the trainer.

        Args:
            patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 2.
            min_delta (float, optional): The minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.01.
        """
        self.__early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        return self

    def has_early_stopper(self) -> bool:
        """Check if the trainer has an early stopper.

        Returns:
            bool: True if the trainer has an early stopper, False otherwise.
        """
        return isinstance(self.__early_stopper, EarlyStopper)
