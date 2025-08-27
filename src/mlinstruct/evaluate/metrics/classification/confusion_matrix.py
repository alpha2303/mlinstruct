from typing import List, Self, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from .base_metrics import BaseMetrics
from ..metric_utils import MetricUtils
from ...plots.cm_plotter import DEFAULT_CMAP, ConfusionMatrixPlotter
from ....utils.exception import IncompatibleDimsException, IncompatibleValuesException


def __compute_confusion_matrix(
    truth_array: np.ndarray, pred_array: np.ndarray, class_count: int
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        truth_array (numpy.ndarray): The ground truth (correct) labels.
        pred_array (numpy.ndarray): The predicted labels.
        class_count (int): The number of classes.
    """
    confusion_matrix: np.ndarray = np.zeros((class_count, class_count)).astype(int)

    for i in range(len(truth_array)):
        confusion_matrix[truth_array[i], pred_array[i]] += 1

    return confusion_matrix


class ConfusionMatrix(BaseMetrics):
    """
    Confusion Matrix for classification tasks.

    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix.
        class_labels (list, optional): The class labels.
    """

    def __init__(
        self, confusion_matrix: np.ndarray, class_labels: Optional[List[str]] = None
    ):
        self.__confusion_matrix: np.ndarray = confusion_matrix
        self.__class_labels: Optional[List[str]] = class_labels

    @classmethod
    def from_predictions(
        cls, y: np.ndarray, y_pred: np.ndarray, class_labels: Optional[list] = None
    ) -> Self:
        """
        Create an instance of the ConfusionMatrix from the true and predicted values.

        Args:
            y (numpy.ndarray): The true labels.
            y_pred (numpy.ndarray): The predicted labels.
            class_labels (list, optional): The class labels.

        Returns:
            ConfusionMatrix: The created ConfusionMatrix instance.

        Raises:
            IncompatibleDimsException: If the dimensions of y and y_pred do not match.
            IncompatibleValuesException: If the values in y and y_pred are not compatible.
            Exception: If an unexpected error occurs.
        """
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
        """
        Get the confusion matrix as a NumPy array.

        Rows represent the true classes, and columns represent the predicted classes.

        Returns:
            numpy.ndarray: The confusion matrix.
        """
        return self.__confusion_matrix

    def plot(
        self,
        title: str = "Confusion Matrix",
        xaxis_name: str = "Predicted",
        yaxis_name: str = "True",
        ax: Optional[Axes] = None,
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
            conf_matrix=self.__confusion_matrix,
            class_labels=self.__class_labels,
            **kwargs,
        )
