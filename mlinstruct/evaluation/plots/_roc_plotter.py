import numpy as np
import matplotlib.axes as axes


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
        ax: axes.Axes,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float = None,
    ):
        self._ax = ax
        self._fpr = fpr
        self._tpr = tpr
        self._auc = auc

    def plot(
        self,
        curve_color: str = "darkorange",
        baseline_color: str = "navy",
        title: str = "Receiver operating characteristic (ROC) curve",
        plot_label: str = "ROC Curve",
        xaxis_name: str = "False Positive Rate",
        yaxis_name: str = "True Positive Rate",
        add_legend: bool = True,
        legend_loc: str = "lower right",
    ) -> axes.Axes:
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
            color=curve_color,
            lw=2,
            label=(plot_label + auc_label),
        )
        self._ax.plot([0, 1], [0, 1], color=baseline_color, lw=2, linestyle="--")
        self._ax.set_xlim([0.0, 1.0])
        self._ax.set_ylim([0.0, 1.05])
        self._ax.set_xlabel(xaxis_name)
        self._ax.set_ylabel(yaxis_name)
        self._ax.set_title(title)

        if add_legend:
            self._ax.legend(loc=legend_loc)

        return self._ax
