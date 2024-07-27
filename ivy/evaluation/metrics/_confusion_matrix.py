import numpy as np
import matplotlib.pyplot as plt
from typing import Self
from matplotlib.axes import Axes

from ..plots.confusion_matrix import (
    ConfusionMatrixPlotConfig,
    ConfusionMatrixPlotter,
    DEFAULT_CMP_CONFIG,
)
from ...utils import Result


def _is_valid_input_dimensions(truth_array: np.ndarray, pred_array: np.ndarray) -> bool:
    return truth_array.ndim == 1 and pred_array.ndim == 1


def _is_valid_input_values(truth_array: np.ndarray, pred_array: np.ndarray) -> bool:
    print(len(np.setdiff1d(truth_array, pred_array)))
    return len(np.setdiff1d(truth_array, pred_array)) == 0


class ConfusionMatrix:
    def __init__(
        self,
        confusion_matrix: np.ndarray,
        class_labels: np.ndarray | list = None,
    ):
        self._confusion_matrix = confusion_matrix
        self._class_labels = class_labels

    @classmethod
    def from_predictions(
        cls,
        y: np.ndarray,
        y_pred: np.ndarray,
        class_labels: np.ndarray | list = None,
    ) -> Result[Self, Exception]:
        if not _is_valid_input_dimensions(y, y_pred):
            return Result.err(IncompatibleDimsException(y.shape, y_pred.shape))

        if not _is_valid_input_values(y, y_pred):
            return Result.err(IncompatibleValuesException())

        conf_matrix_size = len(np.unique(y))

        if class_labels is None:
            class_labels = np.arange(conf_matrix_size)

        cm = cls(None, class_labels)

        return Result.ok(cm._compute(y, y_pred))

    def plot(
        self,
        config: ConfusionMatrixPlotConfig = DEFAULT_CMP_CONFIG,
        show_accuracy: bool = True,
    ) -> Result[Axes, Exception]:
        _, ax = plt.subplots()

        if self._confusion_matrix is None:
            return Result.err(Exception("Cannot find Confusion Matrix in memory."))

        accuracy: float = self._accuracy() if show_accuracy else None

        return Result.ok(
            ConfusionMatrixPlotter(
                ax, self._confusion_matrix, self._class_labels, accuracy
            ).plot(config)
        )

    def _compute(
        self,
        truth_array: np.ndarray,
        pred_array: np.ndarray,
    ) -> Self:

        class_count: int = len(self._class_labels)
        self._confusion_matrix: np.ndarray = np.zeros(
            (class_count, class_count)
        ).astype(int)

        for i in range(len(truth_array)):
            self._confusion_matrix[truth_array[i], pred_array[i]] += 1

        return self

    def _accuracy(self) -> float:
        class_count: int = len(self._class_labels)
        pos_pred_count = 0.0

        for i in range(class_count):
            pos_pred_count += self._confusion_matrix[i, i]

        return (pos_pred_count / self._confusion_matrix.sum()) * 100


class IncompatibleDimsException(Exception):
    def __init__(self, shape_1: tuple[int], shape_2: tuple[int]):
        self.message = f"Incompatible Dimensions: {shape_1}, {shape_2}. Size and shape of input arrays must match."
        super(IncompatibleDimsException, self).__init__(self.message)


class IncompatibleValuesException(Exception):
    def __init__(
        self,
    ):
        self.message = (
            "Incompatible Values: Input arrays do not have the same unique values."
        )
        super(IncompatibleValuesException, self).__init__(self.message)
