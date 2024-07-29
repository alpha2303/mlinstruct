from matplotlib import pyplot as plt
import numpy as np
import matplotlib.axes as axes
from matplotlib.colors import Colormap

from ...utils import Option, Result


class ConfusionMatrixPlotConfig:
    def __init__(
        self,
        cmap: Colormap,
        title: str,
        xaxis_name: str,
        yaxis_name: str,
    ):
        self.cmap = cmap
        self.title = title
        self.xaxis_name = xaxis_name
        self.yaxis_name = yaxis_name


DEFAULT_CMP_CONFIG = ConfusionMatrixPlotConfig(
    cmap=plt.cm.Blues,
    title="Confusion Matrix",
    xaxis_name="Predicted",
    yaxis_name="True",
)


class ConfusionMatrixPlotter:
    """
    Creates a plotting object to visualize a Confusion Matrix.

    Required Arguments:
    ax: `matplotlib.axes.Axes` - Matplotlib Axes object on which the plot will be drawn.
    conf_matrix: `numpy.ndarray` - NumPy array representing the input confusion matrix.

    Optional Arguments:
    class_labels: `Option[list]` - List of strings containing the class labels represented in the confusion matrix. Default = `None`
    """

    def __init__(
        self,
        ax: axes.Axes,
        conf_matrix: np.ndarray,
        class_labels: Option[list] = Option.none(),
        # accuracy: float | None = None,
    ):
        self._ax: axes.Axes = ax
        self._conf_matrix: np.ndarray = conf_matrix
        self._class_labels: np.ndarray | list = class_labels.unwrap()
        if class_labels.is_none():
            self._class_labels = np.arange(self._conf_matrix.shape[0])
            
        # self._accuracy = accuracy

    def plot(self, config=DEFAULT_CMP_CONFIG) -> Result[axes.Axes, Exception]:
        """
        plot() -> Generates the Confusion Matrix Heatmap Plot on `matplotlib.axes.Axes` object provided.
        Arguments follow the options provided by Matplotlib.

        Arguments:
        cmap: `matplotlib.colors.ColorMap` - Color palette for the Confusion Matrix heatmap. Default = `matplotlib.pyplot.cm.Blues`
        title: `str` - Title of the plot. Default = `"Confusion Matrix"`
        xaxis_name: `str` - Label of the X axis of the plot. Default = `"True"`
        yaxis_name: `str` - Label of the Y axis of the plot. Default = `"Predicted"`

        """

        if self._conf_matrix is None:
            return Result.err(Exception("Confusion Matrix not initialized."))

        if self._conf_matrix.shape[0] != self._conf_matrix.shape[1]:
            return Result.err(
                Exception(
                    f"Invalid dimensions: {self._conf_matrix.shape}. Square matrix required."
                )
            )

        if len(self._class_labels) != self._conf_matrix.shape[0]:
            return Result.err(
                Exception(
                    f"Number of class labels ({len(self._class_labels)}) do not match length of confusion matrix ({self._conf_matrix.shape[0]})."
                )
            )

        self._ax.matshow(self._conf_matrix, cmap=config.cmap)
        self._ax.set_xlabel(config.xaxis_name)
        self._ax.set_ylabel(config.yaxis_name)
        self._ax.set_title(config.title)
        self._ax.tick_params(
            axis="x", bottom=True, top=False, labelbottom=True, labeltop=False
        )

        if self._class_labels is not None:
            self._ax.set_xticks(
                np.arange(len(self._class_labels)), labels=self._class_labels
            )
            self._ax.set_yticks(
                np.arange(len(self._class_labels)), labels=self._class_labels
            )

        for i in range(self._conf_matrix.shape[0]):
            for j in range(self._conf_matrix.shape[1]):
                self._ax.text(
                    j,
                    i,
                    self._conf_matrix[i, j],
                    va="center",
                    ha="center",
                    color=self._get_text_color(i, j),
                )

        self._ax.figure.subplots_adjust(right=1.0)

        # if self._accuracy:
        #     self._ax.annotate(
        #         text="Accuracy: %0.2f" % self._accuracy,
        #         xy=(75, 13),
        #         xycoords="figure pixels",
        #     )

        return self._ax

    def _get_text_color(self, row_idx, col_idx):
        max_val = self._conf_matrix.max()
        color = "black"
        if max_val > 0 and self._conf_matrix[row_idx, col_idx] / max_val > 0.5:
            color = "white"
        return color
