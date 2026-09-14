from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from mlinstruct.evaluate.metrics.classification import ConfusionMatrix
from mlinstruct.utils.exception import IncompatibleDimsException, IncompatibleValuesException


class TestConfusionMatrix(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.y: np.ndarray = np.array([2, 2, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 1, 0, 2, 2, 1, 0, 0, 0, 0, 2, 1, 1, 2, 2, 2, 1, 0, 1, 0, 0, 0])  # noqa: E501  # fmt: skip
        cls.y_pred: np.ndarray = np.array([2, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2])  # noqa: E501  # fmt: skip
        cls.y_pred_invalid_dims: np.ndarray = np.array([[2, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2], [2, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2]])  # noqa: E501  # fmt: skip
        cls.y_pred_invalid_values: np.ndarray = np.array([-1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2])  # noqa: E501  # fmt: skip
        cls.confusion_matrix: np.ndarray = np.array([[8, 6, 8], [4, 4, 5], [4, 4, 7]])

    def test_get_valid_cm_as_array(self) -> None:
        cm: ConfusionMatrix = ConfusionMatrix(self.confusion_matrix)
        self.assertTrue(np.array_equal(cm.as_ndarray(), self.confusion_matrix))  # type: ignore

    def test_construct_cm_from_predictions(self) -> None:
        cm: ConfusionMatrix = ConfusionMatrix.from_predictions(self.y, self.y_pred)
        self.assertTrue(
            np.array_equal(
                cm.as_ndarray(),
                self.confusion_matrix,  # type: ignore
            )
        )

    def test_construct_cm_from_predictions_invalid_dims_fail(self) -> None:
        with self.assertRaises(IncompatibleDimsException):
            ConfusionMatrix.from_predictions(self.y, self.y_pred_invalid_dims)

    def test_cm_construct_from_predictions_invalid_values_fail(self) -> None:
        with self.assertRaises(IncompatibleValuesException):
            ConfusionMatrix.from_predictions(self.y, self.y_pred_invalid_values)

    def test_cm_construct_from_predictions_y_empty_fail(self) -> None:
        with self.assertRaises(IncompatibleDimsException):
            ConfusionMatrix.from_predictions(np.array([]), self.y_pred)

    def test_cm_construct_from_predictions_ypred_empty_fail(self) -> None:
        with self.assertRaises(IncompatibleDimsException):
            ConfusionMatrix.from_predictions(self.y, np.array([]))

    def test_cm_construct_from_predictions_both_empty_fail(self) -> None:
        with self.assertRaises(IncompatibleValuesException):
            ConfusionMatrix.from_predictions(np.array([]), np.array([]))

    def test_missing_predicted_class_is_valid(self) -> None:
        y: np.ndarray = np.array([0, 1, 2, 0, 1, 2])
        y_pred: np.ndarray = np.array([0, 1, 1, 0, 1, 1])

        cm: ConfusionMatrix = ConfusionMatrix.from_predictions(y, y_pred)

        self.assertEqual(cm.as_ndarray().shape, (3, 3))
        self.assertEqual(cm.as_ndarray()[2].sum(), 2)

    def test_num_classes_override_pads_matrix(self) -> None:
        y: np.ndarray = np.array([0, 1])
        y_pred: np.ndarray = np.array([0, 1])

        cm: ConfusionMatrix = ConfusionMatrix.from_predictions(y, y_pred, num_classes=5)

        self.assertEqual(cm.as_ndarray().shape, (5, 5))

    def test_non_contiguous_labels(self) -> None:
        y: np.ndarray = np.array([0, 2, 4])
        y_pred: np.ndarray = np.array([0, 2, 4])

        cm: ConfusionMatrix = ConfusionMatrix.from_predictions(y, y_pred)

        expected = np.zeros((5, 5), dtype=int)
        expected[0, 0] = expected[2, 2] = expected[4, 4] = 1
        self.assertTrue(np.array_equal(cm.as_ndarray(), expected))

    def test_plot_creates_axes_when_none_given(self) -> None:
        cm: ConfusionMatrix = ConfusionMatrix(self.confusion_matrix, class_labels=["A", "B", "C"])

        ax = cm.plot()

        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_title(), "Confusion Matrix")

    def test_plot_uses_given_axes(self) -> None:
        cm: ConfusionMatrix = ConfusionMatrix(self.confusion_matrix, class_labels=["A", "B", "C"])
        _, given_ax = plt.subplots()

        returned_ax = cm.plot(ax=given_ax)

        self.assertIs(returned_ax, given_ax)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_sklearn_confusion_matrix(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, size=100)
    y_pred = rng.integers(0, 4, size=100)

    cm = ConfusionMatrix.from_predictions(y, y_pred)

    assert np.array_equal(cm.as_ndarray(), sklearn_confusion_matrix(y, y_pred, labels=range(4)))
