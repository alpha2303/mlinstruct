import numpy as np
import matplotlib.pyplot as plt
from typing import Self
from matplotlib.axes import Axes

from ..plots.confusion_matrix import (
    ConfusionMatrixPlotConfig,
    ConfusionMatrixPlotter,
    DEFAULT_CMP_CONFIG,
)
from ...utils import Option, Result


def _is_valid_input_dimensions(truth_array: np.ndarray, pred_array: np.ndarray) -> bool:
    return truth_array.ndim == 1 and np.array_equal(truth_array.shape, pred_array.shape)


def _is_valid_input_values(truth_array: np.ndarray, pred_array: np.ndarray) -> bool:
    return truth_array.shape[0] > 0 and len(np.setxor1d(truth_array, pred_array)) == 0


def _compute_confusion_matrix(
    truth_array: np.ndarray, pred_array: np.ndarray, class_count: int
) -> np.ndarray:
    confusion_matrix: np.ndarray = np.zeros((class_count, class_count)).astype(int)

    for i in range(len(truth_array)):
        confusion_matrix[truth_array[i], pred_array[i]] += 1

    return confusion_matrix


class ConfusionMatrix:
    def __init__(
        self,
        confusion_matrix: np.ndarray,
    ):
        self._confusion_matrix = confusion_matrix

    @classmethod
    def from_predictions(
        cls, y: np.ndarray, y_pred: np.ndarray
    ) -> Result[Self, Exception]:
        if not _is_valid_input_dimensions(y, y_pred):
            return Result.err(IncompatibleDimsException(y.shape, y_pred.shape))

        if not _is_valid_input_values(y, y_pred):
            return Result.err(IncompatibleValuesException())

        conf_matrix_size = len(np.unique(y))

        try:
            confusion_matrix = _compute_confusion_matrix(y, y_pred, conf_matrix_size)
            return Result.ok(cls(confusion_matrix))
        except Exception as e:
            return Result.err(e)

    def as_array(self) -> Option[np.ndarray]:
        if self._confusion_matrix is not None:
            return Option.some(self._confusion_matrix)
        return Option.none()

    def plot(
        self,
        class_labels: Option[list] = Option.none(),
        config: ConfusionMatrixPlotConfig = DEFAULT_CMP_CONFIG,
        # show_accuracy: bool = True,
    ) -> Result[Axes, Exception]:
        _, ax = plt.subplots()
        # accuracy: float = self._accuracy() if show_accuracy else None

        return ConfusionMatrixPlotter(
            ax,
            self._confusion_matrix,
            class_labels,
        ).plot(config)

    # def _accuracy(self) -> float:
    #     class_count: int = len(self._class_labels)
    #     pos_pred_count = 0.0

    #     for i in range(class_count):
    #         pos_pred_count += self._confusion_matrix[i, i]

    #     return (pos_pred_count / self._confusion_matrix.sum()) * 100


class IncompatibleDimsException(Exception):
    def __init__(self, shape_1: tuple[int], shape_2: tuple[int]):
        self.message: str = (
            f"Incompatible Dimensions: {shape_1}, {shape_2}. Size and shape of input arrays must match."
        )
        super().__init__(self.message)


class IncompatibleValuesException(Exception):
    def __init__(
        self,
    ):
        self.message: str = (
            "Incompatible Values: Input arrays may be empty or do not have the same unique values."
        )
        super().__init__(self.message)
