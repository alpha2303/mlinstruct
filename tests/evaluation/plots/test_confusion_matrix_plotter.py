import numpy as np
from unittest import TestCase
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from mlinstruct.utils import Option, Result
from mlinstruct.evaluation.plots._cm_plotter import (
    ConfusionMatrixPlotter,
    # ConfusionMatrixPlotConfig,
    # DEFAULT_CMP_CONFIG,
)


class TestConfusionMatrixPlotter(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_labels: Option[list[str]] = Option.some(["Car", "Bike", "Scooter"])
        cls.confusion_matrix: np.ndarray = np.array([[8, 6, 8], [4, 4, 5], [4, 4, 7]])
        cls.custom_plot_config: dict = {
            "cmap": plt.cm.Blues,
            "title": "Confusion Matrix Test",
            "xaxis_name": "Prediction values",
            "yaxis_name": "True values",
        }

    def setUp(self) -> None:
        _, self.test_ax = plt.subplots()

    def test_plot_construct_none_cm_fail(self) -> None:
        cm_plotter = ConfusionMatrixPlotter(self.test_ax, None, self.class_labels)
        cm_result = cm_plotter.plot()
        self.assertTrue(cm_result.is_err())

    def test_plot_construct_valid_cm_success(self) -> None:
        cm_plotter = ConfusionMatrixPlotter(
            self.test_ax, self.confusion_matrix, self.class_labels
        )
        cm_result = cm_plotter.plot()
        self.assertFalse(cm_result.is_err())
