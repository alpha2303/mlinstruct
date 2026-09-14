from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from mlinstruct.evaluate.plots.cm_plotter import ConfusionMatrixPlotter


class TestConfusionMatrixPlotter(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_labels: list[str] | None = ["Car", "Bike", "Scooter"]
        cls.confusion_matrix: np.ndarray = np.array([[8, 6, 8], [4, 4, 5], [4, 4, 7]])
        cls.custom_plot_config: dict = {
            "cmap": colormaps.get_cmap("Blues"),
            "title": "Confusion Matrix Test",
            "xaxis_name": "Prediction values",
            "yaxis_name": "True values",
        }

    def setUp(self) -> None:
        _, self.test_ax = plt.subplots()

    def test_init_cm_none_fail(self) -> None:
        cm_plotter = ConfusionMatrixPlotter()
        with self.assertRaises(AttributeError):
            cm_plotter.plot(self.test_ax, None, self.class_labels)  # type: ignore

    def test_init_cm_valid_success(self) -> None:
        cm_plotter = ConfusionMatrixPlotter()
        try:
            cm_plotter.plot(self.test_ax, self.confusion_matrix, self.class_labels)
        except Exception as e:
            self.fail(f"Unexpected error occurred: {e}")

    def test_default_class_labels_used_when_none_given(self) -> None:
        cm_plotter = ConfusionMatrixPlotter()

        try:
            cm_plotter.plot(self.test_ax, self.confusion_matrix)
        except Exception as e:
            self.fail(f"Unexpected error occurred: {e}")

        tick_labels = [label.get_text() for label in self.test_ax.get_xticklabels()]
        self.assertEqual(tick_labels, ["0", "1", "2"])

    def test_non_square_matrix_raises_value_error(self) -> None:
        cm_plotter = ConfusionMatrixPlotter()
        non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError):
            cm_plotter.plot(self.test_ax, non_square_matrix)

    def test_mismatched_class_labels_length_raises_value_error(self) -> None:
        cm_plotter = ConfusionMatrixPlotter()

        with self.assertRaises(ValueError):
            cm_plotter.plot(self.test_ax, self.confusion_matrix, ["Only", "Two"])
