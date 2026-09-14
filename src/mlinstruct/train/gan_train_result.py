from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GANTrainResult:
    """The result of a GAN training session.

    Deliberately omits best_val_loss/val_loss_list/best_checkpoint_path/stopped_early
    from TrainResult: a GAN's generator and discriminator losses trade off against
    each other by construction, so there is no non-arbitrary "best" epoch and no
    early-stopping criterion derived from them.

    Args:
        model_name (str): The name of the model.
        model_save_path (pathlib.Path): The path checkpoints were saved to.
        epochs (int): The number of epochs trained.
        g_loss_list (list[float]): Per-epoch average generator loss.
        d_loss_list (list[float]): Per-epoch average discriminator loss.
        metrics_history (dict[str, list[float]]): Per-epoch metric values collected
            from any callback exposing a .history dict. Empty if none were used.
    """

    model_name: str
    model_save_path: Path
    epochs: int
    g_loss_list: list[float]
    d_loss_list: list[float]
    metrics_history: dict[str, list[float]] = field(default_factory=dict)
