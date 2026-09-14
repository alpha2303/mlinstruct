from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import roc_auc_score

from mlinstruct.evaluate.metrics.classification import ROC
from mlinstruct.utils.exception import IncompatibleDimsException


class TestROC(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.y: np.ndarray = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        cls.y_pred: np.ndarray = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.3, 0.9, 0.55, 0.15])

    def test_from_predictions_runs(self) -> None:
        roc: ROC = ROC.from_predictions(self.y, self.y_pred)
        self.assertIsInstance(roc, ROC)

    def test_auc_matches_sklearn_roc_auc_score(self) -> None:
        roc: ROC = ROC.from_predictions(self.y, self.y_pred)
        self.assertAlmostEqual(roc._auc, roc_auc_score(self.y, self.y_pred))

    def test_from_predictions_invalid_dims_raises(self) -> None:
        y_pred_invalid_dims = np.array([self.y_pred, self.y_pred])
        with self.assertRaises(IncompatibleDimsException):
            ROC.from_predictions(self.y, y_pred_invalid_dims)

    def test_plot_creates_axes_when_none_given(self) -> None:
        roc: ROC = ROC.from_predictions(self.y, self.y_pred)

        ax = roc.plot()

        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_title(), "Receiver operating characteristic (ROC) curve")

    def test_plot_uses_given_axes(self) -> None:
        roc: ROC = ROC.from_predictions(self.y, self.y_pred)
        _, given_ax = plt.subplots()

        returned_ax = roc.plot(ax=given_ax)

        self.assertIs(returned_ax, given_ax)
