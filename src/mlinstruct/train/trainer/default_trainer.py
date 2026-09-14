import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mlinstruct.train.callbacks import TrainerCallback
from mlinstruct.train.data_payload.base_data_payload import BaseDataPayload
from mlinstruct.train.model_proxy.base_model_proxy import BaseModelProxy
from mlinstruct.train.train_result import TrainResult
from mlinstruct.train.trainer.base_trainer import DEFAULT_SAVE_PATH, BaseTrainer
from mlinstruct.train.trainer.epoch_loop_trainer import EpochLoopTrainer
from mlinstruct.train.utils.early_stopper import EarlyStopper
from mlinstruct.utils.exception import TrainerError


class DefaultTrainer(BaseTrainer, EpochLoopTrainer):
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
        callbacks (Sequence[TrainerCallback]): Callbacks invoked at the start of
            training, at the end of every epoch, and at the end of training.
            Defaults to none.
        show_progress (bool): Whether to show a tqdm progress bar over epochs.
            Silently does nothing if tqdm isn't installed. Defaults to False.
    """

    def __init__(
        self,
        model_proxy: BaseModelProxy,
        data_payload: BaseDataPayload,
        early_stopper: EarlyStopper | None = None,
        save_dir_path: Path = DEFAULT_SAVE_PATH,
        run_name: str | None = None,
        logger: logging.Logger | None = None,
        callbacks: Sequence[TrainerCallback] = (),
        show_progress: bool = False,
    ) -> None:
        BaseTrainer.__init__(
            self,
            model_proxy=model_proxy,
            data_payload=data_payload,
            early_stopper=early_stopper,
            save_dir_path=save_dir_path,
            run_name=run_name,
        )
        EpochLoopTrainer.__init__(
            self,
            checkpoint_writer=self._checkpoint_writer,
            callbacks=callbacks,
            show_progress=show_progress,
            logger=logger,
        )
        self._resume_from: Path | None = None
        self._best_vloss: float = np.inf
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
        self._resume_from = resume_from
        return EpochLoopTrainer.train(self, max_epochs)

    def _prepare_run(self) -> int:
        self._best_vloss = np.inf
        if self._resume_from is not None:
            return self._model_proxy.load_checkpoint(self._resume_from) + 1
        return 1

    def _run_epoch(self, epoch_index: int) -> tuple[float, float]:
        avg_loss = self._model_proxy.train_one_epoch(self._data_payload.get_train_data())
        avg_vloss = self._model_proxy.validate(self._data_payload.get_val_data())

        self._logger.info(
            f"Epoch {epoch_index}: Training Loss = {avg_loss} | "
            f"Validation Loss = {avg_vloss} | "
            f"Learning Rate = {self._model_proxy.get_lr()}"
        )

        if self._model_proxy.has_scheduler():
            self._model_proxy.scheduler_step(avg_vloss=avg_vloss)

        return avg_loss, avg_vloss

    def _checkpoint_policy(
        self, epoch_index: int, metric_a: float, metric_b: float, is_final_epoch: bool
    ) -> Path | None:
        avg_vloss = metric_b
        if avg_vloss < self._best_vloss:
            self._best_vloss = avg_vloss
            return self._checkpoint_writer.create_checkpoint(  # type: ignore
                self._model_proxy, epoch_index, avg_vloss
            )
        return None

    def _should_stop_early(self, metric_a: float, metric_b: float) -> bool:
        return bool(self._early_stopper and self._early_stopper.early_stop(metric_b))

    def _after_epochs(self) -> None:
        if self._data_payload.has_test_data():
            avg_tloss = self._model_proxy.validate(
                self._data_payload.get_test_data()  # type: ignore
            )
            self._logger.info(f"Average Test Loss: {avg_tloss}")

    def _build_result(
        self,
        model_save_path: Path,
        epochs_completed: int,
        metric_a_list: list[float],
        metric_b_list: list[float],
        metrics_history: dict[str, list[float]],
        best_checkpoint_path: Path | None,
        stopped_early: bool,
    ) -> TrainResult:
        return TrainResult(
            model_name=self._model_proxy.get_model_name(),
            model_save_path=model_save_path,
            epochs=epochs_completed,
            train_loss_list=metric_a_list,
            val_loss_list=metric_b_list,
            best_val_loss=self._best_vloss,
            best_checkpoint_path=best_checkpoint_path,
            stopped_early=stopped_early,
            metrics_history=metrics_history,
        )

    def _validate_trainer_attrs(self) -> None:
        """Validate that all required trainer attributes are initialized.

        Raises:
            TrainerError: If any of the required trainer attributes are not initialized.
        """
        if not isinstance(self._model_proxy, BaseModelProxy):
            raise TrainerError("Model proxy is not provided.")

        if not isinstance(self._data_payload, BaseDataPayload):
            raise TrainerError("Training Data Payload is not provided.")
