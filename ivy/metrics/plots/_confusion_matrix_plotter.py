from matplotlib import pyplot as plt
import numpy as np
import matplotlib.axes as axes
from matplotlib.colors import Colormap


class ConfusionMatrixPlotter:
    """
    Creates a plotting object to visualize a Confusion Matrix.

    Required Arguments:
    ax: `matplotlib.axes.Axes` - Matplotlib Axes object on which the plot will be drawn.
    conf_matrix: `numpy.ndarray` - NumPy array representing the input confusion matrix.

    Optional Arguments:
    class_labels: `list[str]` - List of strings containing the class labels represented in the confusion matrix. Default = `None`
    accuracy: `float` - Accuracy value obtained from the confusion matrix provided. Default = `None`
    """

    def __init__(
        self,
        ax: axes.Axes,
        conf_matrix: np.ndarray,
        class_labels: list[str] = None,
        accuracy: float = None,
    ):
        self._ax = ax
        self._conf_matrix = conf_matrix
        self._class_labels = class_labels
        self._accuracy = accuracy

    def plot(
        self,
        cmap: Colormap = plt.cm.Blues,
        title: str = "Confusion Matrix",
        xaxis_name: str = "True",
        yaxis_name: str = "Predicted",
    ) -> axes.Axes:
        """
        plot() -> Generates the Confusion Matrix Heatmap Plot on `matplotlib.axes.Axes` object provided.
        Arguments follow the options provided by Matplotlib.

        Arguments:
        cmap: `matplotlib.colors.ColorMap` - Color palette for the Confusion Matrix heatmap. Default = `matplotlib.pyplot.cm.Blues`
        title: `str` - Title of the plot. Default = `"Confusion Matrix"`
        xaxis_name: `str` - Label of the X axis of the plot. Default = `"True"`
        yaxis_name: `str` - Label of the Y axis of the plot. Default = `"Predicted"`

        """

        self._ax.matshow(self._conf_matrix, cmap=cmap)
        self._ax.set_xlabel(xaxis_name)
        self._ax.set_ylabel(yaxis_name)
        self._ax.set_title(title)
        self._ax.tick_params(
            axis="x", bottom=True, top=False, labelbottom=True, labeltop=False
        )

        if self._class_labels:
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

        if self._accuracy:
            self._ax.annotate(
                text="Accuracy: %0.2f" % self._accuracy,
                xy=(350, 18),
                xycoords="figure pixels",
            )

        return self._ax

    def _get_text_color(self, row_idx, col_idx):
        max_val = self._conf_matrix.max()
        color = "black"
        if max_val > 0 and self._conf_matrix[row_idx, col_idx] / max_val > 0.5:
            color = "white"
        return color
