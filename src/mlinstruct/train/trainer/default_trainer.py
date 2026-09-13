import logging
from pathlib import Path

import numpy as np

from ...utils.exception import TrainerError
from ..data_payload.base_data_payload import BaseDataPayload
from ..model_proxy.base_model_proxy import BaseModelProxy
from ..train_result import TrainResult
from ..trainer.base_trainer import DEFAULT_SAVE_PATH, BaseTrainer
from ..utils.early_stopper import EarlyStopper


class DefaultTrainer(BaseTrainer):
    """Default implementation of the trainer, for use with PyTorch models.

    Args:
        model_proxy (BaseModelProxy): Proxy of model to be trained.
        data_payload (BaseDataPayload): Data payload containing training, validation,
            and optional test data.
        early_stopper (Optional[EarlyStopper]): Early stopper for stopping training early.
        save_dir_path (Path): Root directory path for saving model checkpoints.
        run_name (Optional[str]): Name for this run's checkpoint directory.
            Defaults to a timestamp; a name that already exists under
            save_dir_path gets a numeric suffix.
        logger (Optional[logging.Logger]): Logger for logging training progress.
            Defaults to the module logger.
    """

    def __init__(
        self,
        model_proxy: BaseModelProxy,
        data_payload: BaseDataPayload,
        early_stopper: EarlyStopper | None = None,
        save_dir_path: Path = DEFAULT_SAVE_PATH,
        run_name: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            model_proxy=model_proxy,
            data_payload=data_payload,
            early_stopper=early_stopper,
            save_dir_path=save_dir_path,
            run_name=run_name,
        )
        self._logger: logging.Logger = logger or logging.getLogger(__name__)
        self._validate_trainer_attrs()

    def train(self, max_epochs: int, resume_from: Path | None = None) -> TrainResult:
        """Train the model.

        Args:
            max_epochs (int): The maximum epoch to train up to. Remains the
                absolute upper bound even when resuming.
            resume_from (Optional[Path]): Path to a checkpoint to resume from.
                Training starts at the checkpoint's epoch + 1.

        Returns:
            TrainResult: The result of the training process.

        Raises:
            InitException: If any of the required trainer attributes are not initialized.
        """

        if max_epochs <= 0:
            raise ValueError("Max epochs must be positive number greater than 0.")

        best_vloss: float = np.inf
        best_checkpoint_path: Path | None = None
        stopped_early: bool = False
        train_loss_list, val_loss_list = [], []

        self._checkpoint_writer.regenerate_model_save_path()  # type: ignore

        start_epoch: int = 1
        if resume_from is not None:
            start_epoch = self._model_proxy.load_checkpoint(resume_from) + 1

        epochs_completed: int = start_epoch - 1
        try:
            for epoch_index in range(start_epoch, max_epochs + 1):
                avg_loss = self._model_proxy.train_one_epoch(self._data_payload.get_train_data())

                avg_vloss = self._model_proxy.validate(self._data_payload.get_val_data())

                self._logger.info(
                    f"Epoch {epoch_index}: Training Loss = {avg_loss} | "
                    f"Validation Loss = {avg_vloss} | "
                    f"Learning Rate = {self._model_proxy.get_lr()}"
                )

                train_loss_list.append(avg_loss)
                val_loss_list.append(avg_vloss)

                if self._model_proxy.has_scheduler():
                    self._model_proxy.scheduler_step(avg_vloss=avg_vloss)

                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss

                    best_checkpoint_path = self._checkpoint_writer.create_checkpoint(
                        self._model_proxy, epoch_index, avg_vloss
                    )

                epochs_completed = epoch_index

                if self._early_stopper and self._early_stopper.early_stop(avg_vloss):
                    self._logger.info(f"Early stop triggered at epoch: {epoch_index}")
                    stopped_early = True
                    break

            if self._data_payload.has_test_data():
                avg_tloss = self._model_proxy.validate(
                    self._data_payload.get_test_data()  # type: ignore
                )
                self._logger.info(f"Average Test Loss: {avg_tloss}")

            model_save_path = self._checkpoint_writer.get_model_save_path().resolve()  # type: ignore
            self._logger.info(f"Model checkpoints saved to {model_save_path}")

            return TrainResult(
                model_name=self._model_proxy.get_model_name(),
                model_save_path=model_save_path,
                epochs=epochs_completed,
                train_loss_list=train_loss_list,
                val_loss_list=val_loss_list,
                best_val_loss=best_vloss,
                best_checkpoint_path=best_checkpoint_path,
                stopped_early=stopped_early,
            )

        except Exception as e:
            self._logger.error(f"Error during training: {str(e)}")
            raise

    def _validate_trainer_attrs(self) -> None:
        """Validate that all required trainer attributes are initialized.

        Raises:
            TrainerError: If any of the required trainer attributes are not initialized.
        """
        if not isinstance(self._model_proxy, BaseModelProxy):
            raise TrainerError("Model proxy is not provided.")

        if not isinstance(self._data_payload, BaseDataPayload):
            raise TrainerError("Training Data Payload is not provided.")
