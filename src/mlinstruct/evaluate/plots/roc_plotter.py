from typing import Optional
from matplotlib.colors import Colormap
import numpy as np
import matplotlib.axes as axes

from .base_plotter import BasePlotter, DEFAULT_CMAP


class ROCPlotter(BasePlotter):
    """Creates a plotting object to visualize ROC Curve.

    Args:
        title (str, optional): Title of the plot. Defaults to "Receiver operating characteristic (ROC) curve".
        xaxis_name (str, optional): Label for X axis. Defaults to "False Positive Rate".
        yaxis_name (str, optional): Label for Y axis. Defaults to "True Positive Rate".
        curve_color (str, optional): Color of the ROC curve. Defaults to "darkorange".
        baseline_color (str, optional): Color of the baseline. Defaults to "navy".
        plot_label (str, optional): Label for the ROC curve. Defaults to "ROC Curve".
        add_legend (bool, optional): Whether to add legend. Defaults to True.
        legend_loc (str, optional): Location of the legend. Defaults to "lower right".
        cmap (Colormap, optional): Colormap for the plot. Defaults to DEFAULT_CMAP.
    """

    def __init__(
        self,
        title: str = "Receiver operating characteristic (ROC) curve",
        xaxis_name: str = "False Positive Rate",
        yaxis_name: str = "True Positive Rate",
        curve_color: str = "darkorange",
        baseline_color: str = "navy",
        plot_label: str = "ROC Curve",
        add_legend: bool = True,
        legend_loc: str = "lower right",
        cmap: Colormap = DEFAULT_CMAP,
    ):
        super().__init__(title, xaxis_name, yaxis_name, cmap)
        self.__curve_color: str = curve_color
        self.__baseline_color: str = baseline_color
        self.__plot_label: str = plot_label
        self.__add_legend: bool = add_legend
        self.__legend_loc: str = legend_loc

    def plot(
        self,
        ax: axes.Axes,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: Optional[float] = None,
        **kwargs,
    ) -> axes.Axes:
        """Generates the ROC Curve on provided matplotlib axes object.

        Args:
            ax (axes.Axes): Matplotlib Axes object on which to draw the plot.
            fpr (np.ndarray): Array containing false positive rates.
            tpr (np.ndarray): Array containing true positive rates.
            auc (float, optional): Area Under Curve value. Defaults to None.
            **kwargs: Additional keyword arguments for matplotlib plotting.

        Returns:
            axes.Axes: The matplotlib axes object with the plotted ROC curve.
        """
        auc_label = "(AUC = %0.2f)" % auc if auc is not None else ""

        ax.plot(
            fpr,
            tpr,
            color=self.__curve_color,
            lw=2,
            label=(self.__plot_label + auc_label),
        )
        ax.plot([0, 1], [0, 1], color=self.__baseline_color, lw=2, linestyle="--")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel(self.__xaxis_name)
        ax.set_ylabel(self.__yaxis_name)
        ax.set_title(self.__title)

        if self.__add_legend:
            ax.legend(loc=self.__legend_loc)

        return ax
