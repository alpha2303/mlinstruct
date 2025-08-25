from typing import Self, Optional

from matplotlib.colors import Colormap
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from mlinstruct.evaluate.metrics.metric_utils import MetricUtils
from mlinstruct.evaluate.plots.roc_plotter import ROCPlotter, DEFAULT_CMAP
from mlinstruct.utils.exception import IncompatibleDimsException


def __compute_roc_curve(
    truth_array: np.ndarray, pred_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(truth_array, pred_array)


def __compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return np.trapz(tpr, fpr)


class ROC:
    def __init__(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        thresholds: Optional[np.ndarray] = None,
    ):
        self._fpr: np.ndarray = fpr
        self._tpr: np.ndarray = tpr
        self._auc: float = auc
        self._thresholds: Optional[np.ndarray] = thresholds

    @classmethod
    def from_predictions(
        cls,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> Self:
        if not MetricUtils.is_valid_input_dimensions(y, y_pred):
            raise IncompatibleDimsException(y.shape, y_pred.shape)

        try:
            fpr, tpr, thresholds = __compute_roc_curve(y, y_pred)
            auc = __compute_auc(fpr, tpr)
            return cls(fpr, tpr, auc, thresholds)
        except Exception as e:
            raise e

    def plot(
        self,
        title: str = "Receiver operating characteristic (ROC) curve",
        xaxis_name: str = "False Positive Rate",
        yaxis_name: str = "True Positive Rate",
        cmap: Colormap = DEFAULT_CMAP,
        **kwargs,
    ) -> Axes:
        _, ax = plt.subplots()

        return ROCPlotter(
            title=title, xaxis_name=xaxis_name, yaxis_name=yaxis_name
        ).plot(ax, self._fpr, self._tpr, self._auc)
