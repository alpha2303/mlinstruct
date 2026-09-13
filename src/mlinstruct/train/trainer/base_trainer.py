from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Self

from ..data_payload.base_data_payload import BaseDataPayload
from ..model_proxy.base_model_proxy import BaseModelProxy
from ..train_result import TrainResult
from ..utils.early_stopper import EarlyStopper
from ..utils.checkpoint_writer import CheckpointWriter

DEFAULT_SAVE_PATH: Path = Path("./Models")


class BaseTrainer(ABC):
    """Base class for all trainers."""

    _model_proxy: BaseModelProxy
    _data_payload: BaseDataPayload
    _checkpoint_writer: CheckpointWriter
    _root_save_dir_path: Path
    _early_stopper: Optional[EarlyStopper]

    def __init__(
        self,
        model_proxy: BaseModelProxy,
        data_payload: BaseDataPayload,
        early_stopper: Optional[EarlyStopper] = None,
        save_dir_path: Path = DEFAULT_SAVE_PATH,
        run_name: Optional[str] = None,
    ) -> None:
        self._model_proxy: BaseModelProxy = model_proxy
        self._data_payload: BaseDataPayload = data_payload
        self._early_stopper: Optional[EarlyStopper] = early_stopper
        self._root_save_dir_path: Path = save_dir_path
        self._checkpoint_writer: CheckpointWriter = CheckpointWriter(
            self._root_save_dir_path, run_name=run_name
        )

    @abstractmethod
    def train(self, max_epochs: int, resume_from: Optional[Path] = None) -> TrainResult:
        """Train the model.

        Args:
            max_epochs (int): The maximum number of training epochs.
            resume_from (Optional[Path]): Path to a checkpoint to resume from.

        Returns:
            TrainResult: The result of the training process.
        """
        pass

    def add_early_stop(self, patience: int = 2, min_delta: float = 0.01) -> Self:
        """Add early stopping to the trainer.

        Args:
            patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 2.
            min_delta (float, optional): The minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.01.
        """
        self._early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        return self

    def has_early_stopper(self) -> bool:
        """Check if the trainer has an early stopper.

        Returns:
            bool: True if the trainer has an early stopper, False otherwise.
        """
        return isinstance(self._early_stopper, EarlyStopper)
