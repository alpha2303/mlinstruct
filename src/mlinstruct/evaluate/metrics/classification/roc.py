from typing import Self, Optional

import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..base_metrics import BaseMetrics
from ..metric_utils import MetricUtils
from ...plots.roc_plotter import ROCPlotter
from ....utils.exception import IncompatibleDimsException


def __compute_roc_curve(
    truth_array: np.ndarray, pred_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ROC curve.

    Args:
        truth_array (numpy.ndarray): The ground truth (correct) labels.
        pred_array (numpy.ndarray): The predicted labels.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The false positive rate, true positive rate, and thresholds.
    """
    return roc_curve(truth_array, pred_array)


def __compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute the area under the ROC curve (AUC).

    Args:
        fpr (numpy.ndarray): The false positive rate.
        tpr (numpy.ndarray): The true positive rate.

    Returns:
        float: The computed AUC.
    """
    return np.trapz(tpr, fpr)


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
        self.__fpr: np.ndarray = fpr
        self.__tpr: np.ndarray = tpr
        self.__auc: float = auc

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
            Exception: If an unexpected error occurs.
        """
        if not MetricUtils.is_valid_input_dimensions(y, y_pred):
            raise IncompatibleDimsException(y.shape, y_pred.shape)

        try:
            fpr, tpr, thresholds = __compute_roc_curve(y, y_pred)
            auc = __compute_auc(fpr, tpr)
            return cls(fpr, tpr, auc)
        except Exception as e:
            raise e

    def plot(
        self,
        title: str = "Receiver operating characteristic (ROC) curve",
        xaxis_name: str = "False Positive Rate",
        yaxis_name: str = "True Positive Rate",
        ax: Optional[Axes] = None,
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

        return ROCPlotter(
            title=title, xaxis_name=xaxis_name, yaxis_name=yaxis_name
        ).plot(ax, self.__fpr, self.__tpr, self.__auc, **kwargs)
