import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from matplotlib.axes import Axes
from typing import Self

from .._utils import (
    MetricUtils,
    IncompatibleDimsException,
)
from .._base_metric import BaseMetric
from ...plots import ROCPlotter
from ....utils import Option


class ROC(BaseMetric):
    def __init__(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: np.ndarray,
        thresholds: Option[np.ndarray] = Option.none(),
    ):
        self._fpr = fpr
        self._tpr = tpr
        self._auc = auc
        self._thresholds = thresholds

    @classmethod
    def from_predictions(
        cls,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> Self:
        if not MetricUtils.is_valid_input_dimensions(y, y_pred):
            raise IncompatibleDimsException(y.shape, y_pred.shape)

        try:
            fpr, tpr, thresholds = _compute_roc_curve(y, y_pred)
            auc = _compute_auc(fpr, tpr)
            return cls(fpr, tpr, auc, Option.some(thresholds))
        except Exception as e:
            raise e

    def plot(
        self,
        title: str = "Receiver operating characteristic (ROC) curve",
        xaxis_name: str = "False Positive Rate",
        yaxis_name: str = "True Positive Rate",
        **kwargs
    ) -> Axes:
        _, ax = plt.subplots()

        return ROCPlotter(ax, self._fpr, self._tpr, self._auc).plot(
            title=title, xaxis_name=xaxis_name, yaxis_name=yaxis_name, kwargs=kwargs
        )


def _compute_roc_curve(
    truth_array: np.ndarray, pred_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(truth_array, pred_array)


def _compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return np.trapz(tpr, fpr)
