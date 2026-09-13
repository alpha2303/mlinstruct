from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from ....utils.exception import IncompatibleDimsException, IncompatibleValuesException
from ...plots.cm_plotter import DEFAULT_CMAP, ConfusionMatrixPlotter
from ..base_metrics import BaseMetrics
from ..metric_utils import MetricUtils


def _compute_confusion_matrix(
    truth_array: np.ndarray, pred_array: np.ndarray, class_count: int
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        truth_array (numpy.ndarray): The ground truth (correct) labels.
        pred_array (numpy.ndarray): The predicted labels.
        class_count (int): The number of classes.
    """
    confusion_matrix: np.ndarray = np.zeros((class_count, class_count), dtype=int)
    np.add.at(confusion_matrix, (truth_array, pred_array), 1)
    return confusion_matrix


class ConfusionMatrix(BaseMetrics):
    """
    Confusion Matrix for classification tasks.

    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix.
        class_labels (list, optional): The class labels.
    """

    def __init__(self, confusion_matrix: np.ndarray, class_labels: list[str] | None = None):
        self._confusion_matrix: np.ndarray = confusion_matrix
        self._class_labels: list[str] | None = class_labels

    @classmethod
    def from_predictions(
        cls,
        y: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int | None = None,
        class_labels: list | None = None,
    ) -> Self:
        """
        Create an instance of the ConfusionMatrix from the true and predicted values.

        Args:
            y (numpy.ndarray): The true labels.
            y_pred (numpy.ndarray): The predicted labels.
            num_classes (int, optional): The number of classes. Defaults to
                max(y.max(), y_pred.max()) + 1.
            class_labels (list, optional): The class labels.

        Returns:
            ConfusionMatrix: The created ConfusionMatrix instance.

        Raises:
            IncompatibleDimsException: If the dimensions of y and y_pred do not match.
            IncompatibleValuesException: If the values in y and y_pred are not compatible.
        """
        if not MetricUtils.is_valid_input_dimensions(y, y_pred):
            raise IncompatibleDimsException(y.shape, y_pred.shape)

        if y.shape[0] == 0:
            raise IncompatibleValuesException()

        if num_classes is None:
            num_classes = int(max(y.max(), y_pred.max())) + 1

        if not MetricUtils.is_valid_input_values(y, y_pred, num_classes=num_classes):
            raise IncompatibleValuesException()

        confusion_matrix = _compute_confusion_matrix(y, y_pred, num_classes)
        return cls(confusion_matrix, class_labels)

    def as_ndarray(self) -> np.ndarray:
        """
        Get the confusion matrix as a NumPy array.

        Rows represent the true classes, and columns represent the predicted classes.

        Returns:
            numpy.ndarray: The confusion matrix.
        """
        return self._confusion_matrix

    def plot(
        self,
        title: str = "Confusion Matrix",
        xaxis_name: str = "Predicted",
        yaxis_name: str = "True",
        ax: Axes | None = None,
        cmap: Colormap = DEFAULT_CMAP,
        **kwargs,
    ) -> Axes:
        """
        Plot the confusion matrix.

        Args:
            title (str, optional): The title of the plot.
            xaxis_name (str, optional): The name of the x-axis.
            yaxis_name (str, optional): The name of the y-axis.
            ax (matplotlib.axes.Axes, optional): The axes to plot on.
            cmap (matplotlib.colors.Colormap, optional): The colormap to use.
            **kwargs: Additional keyword arguments to pass to the plotter.
        """
        if ax is None:
            _, ax = plt.subplots()

        return ConfusionMatrixPlotter(
            title=title, xaxis_name=xaxis_name, yaxis_name=yaxis_name, cmap=cmap
        ).plot(
            ax=ax,
            conf_matrix=self._confusion_matrix,
            class_labels=self._class_labels,
            **kwargs,
        )
