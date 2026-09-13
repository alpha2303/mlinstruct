from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .train_result import TrainResult
    from .trainer.base_trainer import BaseTrainer


class TrainerCallback:
    """Base class for trainer callbacks.

    Override any hook to react to training events. The default implementations
    are no-ops, so subclasses only need to override the hooks they care about.
    """

    def on_train_start(self, trainer: "BaseTrainer") -> None:
        """Called once, before the first epoch runs."""
        pass

    def on_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, train_loss: float, val_loss: float
    ) -> None:
        """Called after each epoch completes."""
        pass

    def on_train_end(self, trainer: "BaseTrainer", result: "TrainResult") -> None:
        """Called once, after training finishes (whether or not it stopped early)."""
        pass
