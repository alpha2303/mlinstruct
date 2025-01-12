from typing import Optional
import numpy as np
from unittest import TestCase
import matplotlib.pyplot as plt

from mlinstruct.eval.plots._cm_plotter import ConfusionMatrixPlotter


class TestConfusionMatrixPlotter(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_labels: Optional[list[str]] = ["Car", "Bike", "Scooter"]
        cls.confusion_matrix: np.ndarray = np.array([[8, 6, 8], [4, 4, 5], [4, 4, 7]])
        cls.custom_plot_config: dict = {
            "cmap": plt.cm.Blues,
            "title": "Confusion Matrix Test",
            "xaxis_name": "Prediction values",
            "yaxis_name": "True values",
        }

    def setUp(self) -> None:
        _, self.test_ax = plt.subplots()

    def test_init_cm_none_fail(self) -> None:
        cm_plotter = ConfusionMatrixPlotter(self.test_ax, None, self.class_labels)
        with self.assertRaises(Exception):
            axes = cm_plotter.plot()

    def test_init_cm_valid_success(self) -> None:
        cm_plotter = ConfusionMatrixPlotter(
            self.test_ax, self.confusion_matrix, self.class_labels
        )
        try:
            axes = cm_plotter.plot()
        except:
            self.fail("Unexpected error occurred")
