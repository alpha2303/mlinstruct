from collections.abc import Callable, Iterable

import numpy as np

from ..train.callbacks import TrainerCallback
from .metrics.classification.confusion_matrix import ConfusionMatrix


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that exactly match the true value."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy computed from a ConfusionMatrix's diagonal (correct predictions)."""
    matrix = ConfusionMatrix.from_predictions(y_true, y_pred).as_ndarray()
    return float(matrix.trace() / matrix.sum())


class MetricsCallback(TrainerCallback):
    """Computes metrics against a held-out loader at the end of every epoch.

    Args:
        metrics (dict[str, Callable[[np.ndarray, np.ndarray], float]]): Mapping of
            metric name to a callable(y_true, y_pred) -> float.
        loader: Data passed to the trainer's model proxy's predict method at each
            epoch end.
    """

    def __init__(
        self, metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]], loader: Iterable
    ) -> None:
        self._metrics = metrics
        self._loader = loader
        self.history: dict[str, list[float]] = {name: [] for name in metrics}

    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: float) -> None:
        """Predict over the configured loader and append each metric's value to history."""
        y_true, y_pred = trainer._model_proxy.predict(self._loader)
        for name, metric_fn in self._metrics.items():
            self.history[name].append(metric_fn(y_true, y_pred))
