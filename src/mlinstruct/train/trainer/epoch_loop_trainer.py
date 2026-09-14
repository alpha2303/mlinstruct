import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mlinstruct.train.callbacks import TrainerCallback
from mlinstruct.train.utils.checkpoint_writer import CheckpointWriter


class EpochLoopTrainer(ABC):
    """Shared scaffolding for trainers that run a fixed number of epochs, each
    producing a pair of per-epoch scalar metrics, with checkpointing, an optional
    progress bar, and callback fan-out.

    This is what DefaultTrainer and GANTrainer actually have in common: both run
    max_epochs iterations, each epoch yielding two scalars they log and hand to
    callbacks, optionally checkpoint, and accumulate into two parallel lists for
    the final result. What differs between them is what "one epoch" means (a
    full pass over train+val DataLoaders vs n_critic discriminator steps plus one
    generator step per batch) and how checkpointing/stopping/results are decided
    (best-val-loss + early stopping vs cadence-based, no early stopping) — those
    differences are captured as hooks rather than folded into a shared contract.

    Deliberately does not own model_proxy/data_payload: those are specific to
    the single-model case (see BaseTrainer), while KFoldTrainer orchestrates
    whole other trainers per fold rather than running epochs itself, so neither
    fits this class's contract. It exists alongside BaseTrainer, not beneath or
    above it: DefaultTrainer inherits both; GANTrainer inherits only this one.

    Args:
        checkpoint_writer (CheckpointWriter): Manages the run's checkpoint directory.
            Callers construct this themselves (BaseTrainer already does, for
            DefaultTrainer) so this class never needs to know about save_dir_path/
            run_name.
        callbacks (Sequence[TrainerCallback]): Callbacks invoked at the start of
            training, at the end of every epoch, and at the end of training.
        show_progress (bool): Whether to show a tqdm progress bar over epochs.
            Silently does nothing if tqdm isn't installed. Defaults to False.
        logger (Optional[logging.Logger]): Logger for logging training progress.
            Defaults to a logger named after the concrete trainer's module.
    """

    def __init__(
        self,
        checkpoint_writer: CheckpointWriter,
        callbacks: Sequence[TrainerCallback] = (),
        show_progress: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self._checkpoint_writer = checkpoint_writer
        self._callbacks = callbacks
        self._show_progress = show_progress
        self._logger = logger or logging.getLogger(self.__class__.__module__)

    def _epoch_iterator(self, epochs: range):
        if not self._show_progress:
            return epochs

        try:
            from tqdm import tqdm
        except ImportError:
            return epochs

        return tqdm(epochs, desc="Training")

    def _prepare_run(self) -> int:
        """Reset any per-run state and return the epoch to start counting from.

        Called once, before the first epoch. The default starts at epoch 1;
        override to support resuming (see DefaultTrainer).
        """
        return 1

    def _after_epochs(self) -> None:  # noqa: B027
        """Called once after the epoch loop ends, before results are assembled.

        The default is a no-op; override for work that only makes sense after
        every epoch has run (see DefaultTrainer's held-out test set evaluation).
        """
        pass

    def _should_stop_early(self, metric_a: float, metric_b: float) -> bool:
        """Whether to stop the epoch loop after this epoch's metrics.

        The default never stops early; override to plug in an EarlyStopper or
        similar (see DefaultTrainer).
        """
        return False

    @abstractmethod
    def _run_epoch(self, epoch_index: int) -> tuple[float, float]:
        """Run one epoch's worth of work and return its two scalar metrics."""
        ...

    @abstractmethod
    def _checkpoint_policy(
        self, epoch_index: int, metric_a: float, metric_b: float, is_final_epoch: bool
    ) -> Path | None:
        """Decide whether to checkpoint after this epoch.

        Returns the path written, or None if this epoch wasn't checkpointed.
        """
        ...

    @abstractmethod
    def _build_result(
        self,
        model_save_path: Path,
        epochs_completed: int,
        metric_a_list: list[float],
        metric_b_list: list[float],
        metrics_history: dict[str, list[float]],
        best_checkpoint_path: Path | None,
        stopped_early: bool,
    ) -> Any:
        """Assemble the concrete result dataclass for this trainer."""
        ...

    def train(self, max_epochs: int) -> Any:
        """Run the epoch loop.

        Args:
            max_epochs (int): The maximum epoch to train up to.

        Returns:
            Any: Whatever _build_result assembles (a TrainResult, GANTrainResult, ...).
        """
        if max_epochs <= 0:
            raise ValueError("Max epochs must be positive number greater than 0.")

        self._checkpoint_writer.regenerate_model_save_path()  # type: ignore

        start_epoch = self._prepare_run()
        epochs_completed = start_epoch - 1

        metric_a_list: list[float] = []
        metric_b_list: list[float] = []
        best_checkpoint_path: Path | None = None
        stopped_early = False

        for callback in self._callbacks:
            callback.on_train_start(self)

        try:
            for epoch_index in self._epoch_iterator(range(start_epoch, max_epochs + 1)):
                metric_a, metric_b = self._run_epoch(epoch_index)
                metric_a_list.append(metric_a)
                metric_b_list.append(metric_b)

                for callback in self._callbacks:
                    callback.on_epoch_end(self, epoch_index, metric_a, metric_b)

                epochs_completed = epoch_index

                checkpoint_path = self._checkpoint_policy(
                    epoch_index, metric_a, metric_b, epoch_index == max_epochs
                )
                if checkpoint_path is not None:
                    best_checkpoint_path = checkpoint_path

                if self._should_stop_early(metric_a, metric_b):
                    self._logger.info(f"Early stop triggered at epoch: {epoch_index}")
                    stopped_early = True
                    break

            self._after_epochs()

            model_save_path = self._checkpoint_writer.get_model_save_path().resolve()  # type: ignore
            self._logger.info(f"Model checkpoints saved to {model_save_path}")

            metrics_history: dict[str, list[float]] = {}
            for callback in self._callbacks:
                if hasattr(callback, "history"):
                    metrics_history.update(callback.history)

            result = self._build_result(
                model_save_path,
                epochs_completed,
                metric_a_list,
                metric_b_list,
                metrics_history,
                best_checkpoint_path,
                stopped_early,
            )

            for callback in self._callbacks:
                callback.on_train_end(self, result)

            return result

        except Exception as e:
            self._logger.error(f"Error during training: {str(e)}")
            raise
