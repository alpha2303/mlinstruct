from pathlib import Path
from typing import Optional, Self, Tuple
import numpy as np

from train.data_payload.base_data_payload import BaseDataPayload
from train.model_proxy.base_model_proxy import BaseModelProxy
from train.utils import EarlyStopper, CheckpointWriter

_DEFAULT_SAVE_PATH: Path = Path("./Models")


class BaseTrainer:
    """Base class for all trainers."""

    __model_proxy: BaseModelProxy
    __data_payload: BaseDataPayload
    __early_stopper: Optional[EarlyStopper] = None
    __checkpoint_writer: Optional[CheckpointWriter] = None

    def __init__(self, model_proxy: BaseModelProxy, data_payload: BaseDataPayload) -> None:
        self.__model_proxy = model_proxy
        self.__data_payload = data_payload

    def train(self, max_epochs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Train the model.

        Args:
            max_epochs (int): The maximum number of training epochs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The training and validation losses.
        """
        raise NotImplementedError()

    def add_early_stop(self, patience: int = 2, min_delta: float = 0.01) -> Self:
        """Add early stopping to the trainer.

        Args:
            patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 2.
            min_delta (float, optional): The minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.01.
        """
        self.__early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        return self

    def add_checkpoint_save(
        self, save_root_dir_path: Path = _DEFAULT_SAVE_PATH
    ) -> Self:
        """
        Add a checkpoint writer to the trainer.

        Args:
            save_root_dir_path (Path, optional): The root directory path to save checkpoints. Defaults to "./Models".
        """
        self.__checkpoint_writer = CheckpointWriter(save_root_dir_path)
        return self

    def has_checkpoint_writer(self) -> bool:
        """Check if the trainer has a checkpoint writer.

        Returns:
            bool: True if the trainer has a checkpoint writer, False otherwise.
        """
        return self.__checkpoint_writer is not None and isinstance(
            self.__checkpoint_writer, CheckpointWriter
        )

    def has_early_stopper(self) -> bool:
        """Check if the trainer has an early stopper.

        Returns:
            bool: True if the trainer has an early stopper, False otherwise.
        """
        return self.__early_stopper is not None and isinstance(
            self.__early_stopper, EarlyStopper
        )
