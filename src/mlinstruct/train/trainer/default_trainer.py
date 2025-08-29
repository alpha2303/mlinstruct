import logging
from pathlib import Path
import numpy as np
from typing import Optional

from ..data_payload.base_data_payload import BaseDataPayload
from ..data.train_result import TrainResult
from ..data.enum import ModelFormat
from ..model_proxy.base_model_proxy import BaseModelProxy
from ..trainer.base_trainer import DEFAULT_SAVE_PATH, BaseTrainer
from ..utils.checkpoint_writer import CheckpointWriter
from ..utils.early_stopper import EarlyStopper
from ...utils.exception import TrainerError


class DefaultTrainer(BaseTrainer):
    """Default implementation of the trainer, for use with PyTorch models.

    Args:
        model_proxy (BaseModelProxy): Proxy of model to be trained.
        data_payload (BaseDataPayload): Data payload containing training, validation, and optional test data.
        early_stopper (Optional[EarlyStopper]): Early stopper for stopping training early.
        save_dir_path (Path): Root directory path for saving model checkpoints.
        logger (logging.Logger): Logger for logging training progress.
    """

    def __init__(
        self,
        model_proxy: BaseModelProxy,
        data_payload: BaseDataPayload,
        early_stopper: Optional[EarlyStopper] = None,
        save_dir_path: Path = DEFAULT_SAVE_PATH,
        save_format: ModelFormat = ModelFormat.PT,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.__model_proxy: BaseModelProxy = model_proxy
        self.__data_payload: BaseDataPayload = data_payload
        self.__early_stopper: Optional[EarlyStopper] = early_stopper
        self.__root_save_dir_path: Path = save_dir_path
        self.__checkpoint_writer: CheckpointWriter = CheckpointWriter(
            self.__root_save_dir_path
        )
        self.__save_format: ModelFormat = save_format
        self.__logger: logging.Logger = logger
        self.__validate_trainer_attrs()

    def train(self, max_epochs: int) -> TrainResult:
        """Train the model.

        Args:
            max_epochs (int): The maximum number of training epochs.

        Returns:
            TrainResult: The result of the training process.

        Raises:
            InitException: If any of the required trainer attributes are not initialized.
        """

        if max_epochs <= 0:
            raise ValueError("Max epochs must be positive number greater than 0.")

        best_vloss: float = np.inf
        train_loss_list, val_loss_list = [], []

        self.__checkpoint_writer.regenerate_model_save_path()  # type: ignore

        epochs_completed: int = 0
        try:
            for epoch_index in range(1, max_epochs + 1):
                avg_loss = self.__model_proxy.train_one_epoch(
                    self.__data_payload.get_train_data()
                )

                avg_vloss = self.__model_proxy.validate(
                    self.__data_payload.get_val_data()
                )

                self.__logger.info(
                    f"Epoch {epoch_index}: Training Loss = {avg_loss} | Validation Loss = {avg_vloss} | Learning Rate = {self.__model_proxy.get_lr()}"
                )

                train_loss_list.append(avg_loss)
                val_loss_list.append(avg_vloss)

                if self.__model_proxy.has_scheduler():
                    self.__model_proxy.scheduler_step(avg_vloss=avg_loss)

                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss

                    self.__checkpoint_writer.create_checkpoint(
                        self.__model_proxy, epoch_index, avg_vloss, self.__save_format
                    )

                epochs_completed = epoch_index

                if self.__early_stopper and self.__early_stopper.early_stop(avg_vloss):
                    self.__logger.info(f"Early stop triggered at epoch: {epoch_index}")
                    break

            if self.__data_payload.has_test_data():
                avg_tloss = self.__model_proxy.validate(
                    self.__data_payload.get_test_data()  # type: ignore
                )
                self.__logger.info(f"Average Test Loss: {avg_tloss}")

            self.__logger.info(
                f"Model checkpoints saved to {self.__checkpoint_writer.get_model_save_path().resolve()}"  # type: ignore
            )

            return TrainResult(
                model_name=self.__model_proxy.get_model_name(),
                model_save_path=self.__checkpoint_writer.get_model_save_path().resolve(),  # type: ignore
                epochs=epochs_completed,
                train_loss_list=train_loss_list,
                val_loss_list=val_loss_list,
            )

        except Exception as e:
            self.__logger.error(f"Error during training: {str(e)}")
            raise

    def __validate_trainer_attrs(self) -> None:
        """Validate that all required trainer attributes are initialized.

        Raises:
            TrainerError: If any of the required trainer attributes are not initialized.
        """
        if not isinstance(self.__model_proxy, BaseModelProxy):
            raise TrainerError("Model proxy is not provided.")

        if not isinstance(self.__data_payload, BaseDataPayload):
            raise TrainerError("Training Data Payload is not provided.")
