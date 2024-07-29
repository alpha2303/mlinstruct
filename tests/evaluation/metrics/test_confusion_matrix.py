import numpy as np
from unittest import TestCase

from ivy.evaluation.metrics import (
    ConfusionMatrix,
    IncompatibleDimsException,
    IncompatibleValuesException,
)
from ivy.utils import Result


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
        self.assertFalse(cm.as_array().is_none())
        self.assertTrue(np.array_equal(cm.as_array().unwrap(), self.confusion_matrix))

    def test_construct_cm_from_predictions(self) -> None:
        cm_result: Result[ConfusionMatrix, Exception] = (
            ConfusionMatrix.from_predictions(self.y, self.y_pred)
        )
        self.assertFalse(cm_result.is_err())
        self.assertTrue(
            np.array_equal(
                cm_result.unwrap().as_array().unwrap(), self.confusion_matrix
            )
        )

    def test_construct_cm_from_predictions_invalid_dims_fail(self) -> None:
        cm_result: Result[ConfusionMatrix, IncompatibleDimsException] = (
            ConfusionMatrix.from_predictions(self.y, self.y_pred_invalid_dims)
        )
        self.assertTrue(cm_result.is_err())
        self.assertTrue(isinstance(cm_result.unwrap(), IncompatibleDimsException))

    def test_cm_construct_from_predictions_invalid_values_fail(self) -> None:
        cm_result: Result[ConfusionMatrix, IncompatibleValuesException] = (
            ConfusionMatrix.from_predictions(self.y, self.y_pred_invalid_values)
        )
        self.assertTrue(cm_result.is_err())
        self.assertTrue(isinstance(cm_result.unwrap(), IncompatibleValuesException))

    def test_cm_construct_from_predictions_y_empty_fail(self) -> None:
        cm_result: Result[ConfusionMatrix, IncompatibleDimsException] = (
            ConfusionMatrix.from_predictions(np.array([]), self.y_pred)
        )
        self.assertTrue(cm_result.is_err())
        self.assertTrue(isinstance(cm_result.unwrap(), IncompatibleDimsException))

    def test_cm_construct_from_predictions_ypred_empty_fail(self) -> None:
        cm_result: Result[ConfusionMatrix, IncompatibleDimsException] = (
            ConfusionMatrix.from_predictions(self.y, np.array([]))
        )
        self.assertTrue(cm_result.is_err())
        self.assertTrue(isinstance(cm_result.unwrap(), IncompatibleDimsException))

    def test_cm_construct_from_predictions_both_empty_fail(self) -> None:
        cm_result: Result[ConfusionMatrix, IncompatibleValuesException] = (
            ConfusionMatrix.from_predictions(np.array([]), np.array([]))
        )
        self.assertTrue(cm_result.is_err())
        self.assertTrue(isinstance(cm_result.unwrap(), IncompatibleValuesException))
