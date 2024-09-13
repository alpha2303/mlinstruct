import numpy as np
import matplotlib.pyplot as plt
from typing import Self
from matplotlib.axes import Axes

from .._metric_utils import (
    MetricUtils,
    IncompatibleDimsException,
    IncompatibleValuesException,
)
from ...plots import (
    # ConfusionMatrixPlotConfig,
    ConfusionMatrixPlotter,
    # DEFAULT_CMP_CONFIG,
)
from ....utils import Option, Result

def _compute_confusion_matrix(
    truth_array: np.ndarray, pred_array: np.ndarray, class_count: int
) -> np.ndarray:
    confusion_matrix: np.ndarray = np.zeros((class_count, class_count)).astype(int)

    for i in range(len(truth_array)):
        confusion_matrix[truth_array[i], pred_array[i]] += 1

    return confusion_matrix


class ConfusionMatrix:
    def __init__(
        self, confusion_matrix: np.ndarray, class_labels: Option[list] = Option.none()
    ):
        self._confusion_matrix = confusion_matrix
        self._class_labels = class_labels

    @classmethod
    def from_predictions(
        cls,
        y: np.ndarray,
        y_pred: np.ndarray,
        class_labels: Option[list] = Option.none(),
    ) -> Result[Self, Exception]:

        if not MetricUtils.is_valid_input_dimensions(y, y_pred):
            return Result.err(IncompatibleDimsException(y.shape, y_pred.shape))

        if not MetricUtils.is_valid_input_values(y, y_pred):
            return Result.err(IncompatibleValuesException())

        try:
            confusion_matrix = _compute_confusion_matrix(y, y_pred, len(np.unique(y)))
            return Result.ok(cls(confusion_matrix, class_labels))
        except Exception as e:
            return Result.err(e)

    def as_ndarray(self) -> Option[np.ndarray]:
        if self._confusion_matrix is not None:
            return Option.some(self._confusion_matrix)
        return Option.none()

    def plot(
        self,
        title: str = "Confusion Matrix",
        xaxis_name: str = "Predicted",
        yaxis_name: str = "True",
        **kwargs,
    ) -> Result[Axes, Exception]:
        _, ax = plt.subplots()

        return ConfusionMatrixPlotter(
            ax,
            self._confusion_matrix,
            self._class_labels,
        ).plot(title=title, xaxis_name=xaxis_name, yaxis_name=yaxis_name, kwargs=kwargs)


# def _labelize_binary(y_pred: np.ndarray, threshold: float) -> np.ndarray:
#     return np.where(y_pred > threshold, 1, 0)


# def _labelize_multiclass(y_pred: np.ndarray) -> np.ndarray:
#     return [np.argmax(y_pred[i])[0] for i in y_pred.shape[0]]



