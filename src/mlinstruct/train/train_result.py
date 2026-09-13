from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainResult:
    """The result of a training session.

    Args:
        model_name (str): The name of the model.
        model_save_path (pathlib.Path): The path to save the model.
        epochs (int): The number of epochs trained.
        train_loss_list (List[float]): The list of training losses.
        val_loss_list (List[float]): The list of validation losses.
        best_val_loss (float): The best (lowest) validation loss seen during training.
        best_checkpoint_path (Optional[Path]): The path to the best checkpoint written,
            or None if no checkpoint was written.
        stopped_early (bool): Whether training stopped early via the EarlyStopper.
        metrics_history (dict[str, list[float]]): Per-epoch metric values collected
            from any MetricsCallback-like callbacks. Empty if none were used.
    """

    model_name: str
    model_save_path: Path
    epochs: int
    train_loss_list: list[float]
    val_loss_list: list[float]
    best_val_loss: float
    best_checkpoint_path: Path | None
    stopped_early: bool
    metrics_history: dict[str, list[float]] = field(default_factory=dict)
