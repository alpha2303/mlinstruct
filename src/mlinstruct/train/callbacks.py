from typing import Any


class TrainerCallback:
    """Base class for trainer callbacks.

    Override any hook to react to training events. The default implementations
    are no-ops, so subclasses only need to override the hooks they care about.

    The hooks are typed structurally (Any) rather than to BaseTrainer/TrainResult:
    GANTrainer/GANTrainResult reuse these same hooks by duck typing (see
    GANTrainer's docstring), so pinning the type to one hierarchy would be
    inaccurate for the other.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called once, before the first epoch runs."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, train_loss: float, val_loss: float) -> None:
        """Called after each epoch completes."""
        pass

    def on_train_end(self, trainer: Any, result: Any) -> None:
        """Called once, after training finishes (whether or not it stopped early)."""
        pass
