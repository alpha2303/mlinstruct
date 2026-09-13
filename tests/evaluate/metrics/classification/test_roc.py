from unittest import TestCase

import numpy as np
from sklearn.metrics import roc_auc_score

from mlinstruct.evaluate.metrics.classification import ROC


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
