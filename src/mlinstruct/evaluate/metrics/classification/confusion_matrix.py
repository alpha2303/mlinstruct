from typing import Self, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from ..metric_utils import MetricUtils
from ...plots.base_plotter import DEFAULT_CMAP
from ...plots.cm_plotter import ConfusionMatrixPlotter
from ....utils.exception import IncompatibleDimsException, IncompatibleValuesException


def __compute_confusion_matrix(
    truth_array: np.ndarray, pred_array: np.ndarray, class_count: int
) -> np.ndarray:
    confusion_matrix: np.ndarray = np.zeros((class_count, class_count)).astype(int)

    for i in range(len(truth_array)):
        confusion_matrix[truth_array[i], pred_array[i]] += 1

    return confusion_matrix


class ConfusionMatrix:
    def __init__(
        self, confusion_matrix: np.ndarray, class_labels: Optional[list] = None
    ):
        self._confusion_matrix = confusion_matrix
        self._class_labels = class_labels

    @classmethod
    def from_predictions(
        cls, y: np.ndarray, y_pred: np.ndarray, class_labels: Optional[list] = None
    ) -> Self:
        if not MetricUtils.is_valid_input_dimensions(y, y_pred):
            raise IncompatibleDimsException(y.shape, y_pred.shape)

        if not MetricUtils.is_valid_input_values(y, y_pred):
            raise IncompatibleValuesException()

        try:
            confusion_matrix = __compute_confusion_matrix(y, y_pred, len(np.unique(y)))
            return cls(confusion_matrix, class_labels)
        except Exception as e:
            raise e

    def as_ndarray(self) -> Optional[np.ndarray]:
        if self._confusion_matrix is not None:
            return self._confusion_matrix
        return None

    def plot(
        self,
        title: str = "Confusion Matrix",
        xaxis_name: str = "Predicted",
        yaxis_name: str = "True",
        cmap: Colormap = DEFAULT_CMAP,
    ) -> Axes:
        _, ax = plt.subplots()

        return ConfusionMatrixPlotter(
            title=title, xaxis_name=xaxis_name, yaxis_name=yaxis_name, cmap=cmap
        ).plot(
            ax=ax, conf_matrix=self._confusion_matrix, class_labels=self._class_labels
        )
