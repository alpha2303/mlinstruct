import numpy as np
from matplotlib.axes import Axes

from ...utils import Option


class ROCPlotConfig:
    def __init__(
        self,
        title: str,
        xaxis_name: str,
        yaxis_name: str,
        curve_color: str,
        baseline_color: str,
        plot_label: str,
        add_legend: bool,
        legend_loc: str,
    ):
        self.title: str = title
        self.xaxis_name: str = xaxis_name
        self.yaxis_name: str = yaxis_name
        self.curve_color: str = curve_color
        self.baseline_color: str = baseline_color
        self.plot_label: str = plot_label
        self.add_legend: bool = add_legend
        self.legend_loc: str = legend_loc


DEFAULT_ROCP_CONFIG = ROCPlotConfig(
    title="Receiver operating characteristic (ROC) curve",
    xaxis_name="False Positive Rate",
    yaxis_name="True Positive Rate",
    curve_color="darkorange",
    baseline_color="navy",
    plot_label="ROC Curve",
    add_legend=True,
    legend_loc="lower right",
)


class ROCPlotter:
    """
    Creates a plotting object to visualize ROC Curve.

    Required Arguments:
    ax: `matplotlib.axes.Axes` - Matplotlib Axes object on which the plot will be drawn.
    fpr: `numpy.ndarray` - NumPy array containing the false positive rate (FPR) of each training epoch.
    tpr: `numpy.ndarray` - NumPy array containing the true positive rate (FPR) of each training epoch.

    Optional Arguments:
    auc: `float` - Area Under Curve (AUC) value of the generated ROC curve. Default = `None`
    """

    def __init__(
        self,
        ax: Axes,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float = None,
        plot_config: Option[ROCPlotConfig] = Option.some(DEFAULT_ROCP_CONFIG),
    ):
        self._ax: Axes = ax
        self._fpr: np.ndarray = fpr
        self._tpr: np.ndarray = tpr
        self._auc: float = auc
        self._plot_config: ROCPlotConfig = plot_config.unwrap()

    def plot(self) -> Axes:
        """
        plot() -> Generates the ROC Curve on `matplotlib.axes.Axes` object provided.
        Arguments follow the options provided by Matplotlib.

        Arguments:
        curve_color: `str` - Color of the ROC curve plot. Default = `"darkorange"`
        diagonal_color: `str` - Color of the diagonal baseline (random guess curve). Default = `"navy"`
        title: `str` - Title of the plot. Default = `"Receiver operating characteristic (ROC) curve"`
        xaxis_name: `str` - Label of the X axis of the plot. Default = `"False Positive Rate"`
        yaxis_name: `str` - Label of the Y axis of the plot. Default = `"True Positive Rate"`
        add_legend: `bool` - Boolean value indication whether a legend should be added to the plot. Default = `True`
        legend_loc: `str` - Location of the legend on the plot figure, if added. Default = `"lower right"`

        """
        auc_label = " (AUC = %0.2f)" % self._auc if self._auc else ""

        self._ax.plot(
            self._fpr,
            self._tpr,
            color=self._plot_config.curve_color,
            lw=2,
            label=(self._plot_config.plot_label + auc_label),
        )
        self._ax.plot([0, 1], [0, 1], color=self._plot_config.baseline_color, lw=2, linestyle="--")
        self._ax.set_xlim([0.0, 1.0])
        self._ax.set_ylim([0.0, 1.05])
        self._ax.set_xlabel(self._plot_config.xaxis_name)
        self._ax.set_ylabel(self._plot_config.yaxis_name)
        self._ax.set_title(self._plot_config.title)

        if self._plot_config.add_legend:
            self._ax.legend(loc=self._plot_config.legend_loc)

        return self._ax
