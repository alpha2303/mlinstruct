import numpy as np
from unittest import TestCase
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ivy.utils import Option, Result
from ivy.evaluation.plots.confusion_matrix import (
    ConfusionMatrixPlotter,
    ConfusionMatrixPlotConfig,
    DEFAULT_CMP_CONFIG,
)


class TestConfusionMatrixPlotter(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_labels: Option[list[str]] = Option.some(["Car", "Bike", "Scooter"])
        cls.confusion_matrix: np.ndarray = np.array([[8, 6, 8], [4, 4, 5], [4, 4, 7]])
        cls.custom_plot_config: ConfusionMatrixPlotConfig = ConfusionMatrixPlotConfig(
            plt.cm.Blues, "Confusion Matrix Test", "Prediction values", "True values"
        )

    def setUp(self) -> None:
        _, self.test_ax = plt.subplots()

    def test_plot_construct_none_cm_fail(self) -> None:
        cm_plotter = ConfusionMatrixPlotter(self.test_ax, None, self.class_labels)
        cm_result = cm_plotter.plot()
        self.assertTrue(cm_result.is_err())
    
    def test_plot_construct_valid_cm_success(self) -> None:
        cm_plotter = ConfusionMatrixPlotter(self.test_ax, self.confusion_matrix, self.class_labels)
        cm_result = cm_plotter.plot()
        self.assertFalse(cm_result.is_err())