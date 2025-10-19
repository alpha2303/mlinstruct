import numpy as np
from unittest import TestCase

from mlinstruct.evaluate.metrics.classification import ConfusionMatrix
from mlinstruct.utils.exception import IncompatibleDimsException,IncompatibleValuesException


class TestConfusionMatrix(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.y: np.ndarray = np.array(
            [2, 2, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 1, 0, 2, 2, 1, 0, 0, 0, 0, 2, 1, 1, 2, 2, 2, 1, 0, 1, 0, 0, 0]
        )
        cls.y_pred: np.ndarray = np.array(
            [2, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2]
        )
        cls.y_pred_invalid_dims: np.ndarray = np.array(
            [
                [2, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2],
                [2, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2]
            ]
        )
        cls.y_pred_invalid_values: np.ndarray = np.array(
            [3, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2]
        )
        cls.confusion_matrix: np.ndarray = np.array([[8, 6, 8], [4, 4, 5], [4, 4, 7]])

    def test_get_valid_cm_as_array(self) -> None:
        cm: ConfusionMatrix = ConfusionMatrix(self.confusion_matrix)
        self.assertTrue(np.array_equal(cm.as_ndarray(), self.confusion_matrix)) # type: ignore

    def test_construct_cm_from_predictions(self) -> None:
        cm: ConfusionMatrix = ConfusionMatrix.from_predictions(self.y, self.y_pred)
        self.assertTrue(
            np.array_equal(
                cm.as_ndarray(), self.confusion_matrix # type: ignore
            )
        )

    def test_construct_cm_from_predictions_invalid_dims_fail(self) -> None:
        with self.assertRaises(IncompatibleDimsException):
            cm: ConfusionMatrix = ConfusionMatrix.from_predictions(self.y, self.y_pred_invalid_dims)

    def test_cm_construct_from_predictions_invalid_values_fail(self) -> None:
        with self.assertRaises(IncompatibleValuesException):
            cm: ConfusionMatrix = ConfusionMatrix.from_predictions(self.y, self.y_pred_invalid_values)

    def test_cm_construct_from_predictions_y_empty_fail(self) -> None:
        with self.assertRaises(IncompatibleDimsException):
            cm: ConfusionMatrix = ConfusionMatrix.from_predictions(np.array([]), self.y_pred)

    def test_cm_construct_from_predictions_ypred_empty_fail(self) -> None:
        with self.assertRaises(IncompatibleDimsException):
            cm: ConfusionMatrix = ConfusionMatrix.from_predictions(self.y, np.array([]))

    def test_cm_construct_from_predictions_both_empty_fail(self) -> None:
        with self.assertRaises(IncompatibleValuesException):
            cm: ConfusionMatrix = ConfusionMatrix.from_predictions(np.array([]), np.array([]))
