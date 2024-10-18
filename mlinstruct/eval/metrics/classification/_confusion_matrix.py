from typing import Self

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.axes import Axes

from .._base_metric import BaseMetric
from .._utils import (
    MetricUtils,
    IncompatibleDimsException,
    IncompatibleValuesException,
)
from ...plots import ConfusionMatrixPlotter, ConfusionMatrixPlotConfig
from ....utils import Option


class ConfusionMatrix(BaseMetric):
    def __init__(
        self, confusion_matrix: np.ndarray, class_labels: list[str] | None = None
    ):
        self._confusion_matrix: np.ndarray = confusion_matrix
        self._class_labels: Option[list[str]] = (
            Option.some(class_labels) if class_labels else Option.none()
        )
        self._plot_config: Option[ConfusionMatrixPlotConfig] = Option.none()

    @classmethod
    def from_predictions(cls, y: np.ndarray, y_pred: np.ndarray) -> Self:

        if not MetricUtils.is_valid_input_dimensions(y, y_pred):
            raise IncompatibleDimsException(y.shape, y_pred.shape)

        if not MetricUtils.is_valid_input_values(y, y_pred):
            raise IncompatibleValuesException()

        try:
            confusion_matrix = _compute_confusion_matrix(y, y_pred, len(np.unique(y)))
            return cls(confusion_matrix)
        except Exception as e:
            raise e

    def set_class_labels(self, class_labels: list[str]) -> None:
        self._class_labels = Option.some(class_labels)

    def set_plot_config(
        self,
        title: str = "Confusion Matrix",
        xaxis_name: str = "Predicted",
        yaxis_name: str = "True",
        color_map: Colormap = plt.cm.Blues,
    ) -> None:
        self._plot_config = Option.some(
            ConfusionMatrixPlotConfig(
                cmap=color_map,
                title=title,
                xaxis_name=xaxis_name,
                yaxis_name=yaxis_name,
            )
        )

    def as_ndarray(self) -> Option[np.ndarray]:
        return self._confusion_matrix

    def plot(self) -> Axes:
        _, ax = plt.subplots()

        return ConfusionMatrixPlotter(
            ax, self._confusion_matrix, self._class_labels, self._plot_config
        ).plot()


def _compute_confusion_matrix(
    truth_array: np.ndarray, pred_array: np.ndarray, class_count: int
) -> np.ndarray:
    confusion_matrix: np.ndarray = np.zeros((class_count, class_count)).astype(int)

    for i in range(len(truth_array)):
        confusion_matrix[truth_array[i], pred_array[i]] += 1

    return confusion_matrix


# def _labelize_binary(y_pred: np.ndarray, threshold: float) -> np.ndarray:
#     return np.where(y_pred > threshold, 1, 0)


# def _labelize_multiclass(y_pred: np.ndarray) -> np.ndarray:
#     return [np.argmax(y_pred[i])[0] for i in y_pred.shape[0]]
