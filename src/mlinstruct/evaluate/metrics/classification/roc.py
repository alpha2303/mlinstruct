from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import auc, roc_curve

from mlinstruct.evaluate.metrics.base_metrics import BaseMetrics
from mlinstruct.evaluate.metrics.metric_utils import MetricUtils
from mlinstruct.evaluate.plots.roc_plotter import ROCPlotter
from mlinstruct.utils.exception import IncompatibleDimsException


def _compute_roc_curve(
    truth_array: np.ndarray, pred_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ROC curve.

    Args:
        truth_array (numpy.ndarray): The ground truth (correct) labels.
        pred_array (numpy.ndarray): The predicted labels.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The false positive rate, true positive
            rate, and thresholds.
    """
    return roc_curve(truth_array, pred_array)


def _compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute the area under the ROC curve (AUC).

    Args:
        fpr (numpy.ndarray): The false positive rate.
        tpr (numpy.ndarray): The true positive rate.

    Returns:
        float: The computed AUC.
    """
    return auc(fpr, tpr)


class ROC(BaseMetrics):
    """
    Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (numpy.ndarray): The false positive rate.
        tpr (numpy.ndarray): The true positive rate.
        auc (float): The area under the curve (AUC).
        thresholds (numpy.ndarray, optional): The thresholds used to compute the ROC curve.
    """

    def __init__(self, fpr: np.ndarray, tpr: np.ndarray, auc: float):
        self._fpr: np.ndarray = fpr
        self._tpr: np.ndarray = tpr
        self._auc: float = auc

    @classmethod
    def from_predictions(
        cls,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> Self:
        """
        Create an instance of the ROC from the true and predicted values.

        Args:
            y (numpy.ndarray): The true labels.
            y_pred (numpy.ndarray): The predicted labels.

        Returns:
            ROC: The created ROC instance.

        Raises:
            IncompatibleDimsException: If the dimensions of y and y_pred do not match.
        """
        if not MetricUtils.is_valid_input_dimensions(y, y_pred):
            raise IncompatibleDimsException(y.shape, y_pred.shape)

        fpr, tpr, thresholds = _compute_roc_curve(y, y_pred)
        auc = _compute_auc(fpr, tpr)
        return cls(fpr, tpr, auc)

    def plot(
        self,
        title: str = "Receiver operating characteristic (ROC) curve",
        xaxis_name: str = "False Positive Rate",
        yaxis_name: str = "True Positive Rate",
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes:
        """
        Plot the ROC curve.

        Args:
            title (str, optional): The title of the plot.
            xaxis_name (str, optional): The name of the x-axis.
            yaxis_name (str, optional): The name of the y-axis.
            ax (matplotlib.axes.Axes, optional): The axes to plot on.
            **kwargs: Additional keyword arguments to pass to the plotter.
        """
        if ax is None:
            _, ax = plt.subplots()

        return ROCPlotter(title=title, xaxis_name=xaxis_name, yaxis_name=yaxis_name).plot(
            ax, self._fpr, self._tpr, self._auc, **kwargs
        )
