import numpy as np
import matplotlib.axes as axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap


class TrainingLossPlotter:
    """
    Creates a plotting object to visualize training loss trends.
    Validation loss information can also be included for comparison.

    Required Arguments:
    ax: `matplotlib.axes.Axes` - Matplotlib Axes object on which the plot will be drawn.
    train_losses: `numpy.ndarray` - NumPy array containing the training loss values of each training epoch.

    Optional Arguments:
    val_losses: `numpy.ndarray` - NumPy array containing the validation loss values of each training epoch. Default = `None`
    """
    def __init__(
        self, ax: axes.Axes, train_losses: np.ndarray, val_losses: np.ndarray = None
    ):
        self._ax = ax
        self._train_losses = train_losses
        self._val_losses = val_losses

    def plot(
        self,
        train_color: str = "blue",
        train_label: str = "train",
        val_color: str = "orange",
        val_label: str = "validation",
        title: str = "Training Loss",
        xaxis_name: str = "Epoch",
        yaxis_name: str = "Loss",
        add_legend: bool = True,
        legend_loc: str = "upper right",
    ) -> axes.Axes:
        """
        plot() -> Generates the Training Loss Line Plot on `matplotlib.axes.Axes` object provided.
        Argument values follow values provided by Matplotlib.

        Validation Loss will be plotted only if validation loss values were provided at object creation.

        Arguments:
        train_color: `str` - Color of the training loss plot. Default = `"orange"`
        train_label: `str` - Label of the training loss plot. Default = `"train"`
        val_color: `str` - Color of the validation loss plot. Default = `"orange"`
        val_label: `str` - Label of the validation loss plot. Default = `"train"`
        title: `str` - Title of the plot. Default = `"Training Loss"`
        xaxis_name: `str` - Label of the X axis of the plot. Default = `"Epoch"`
        yaxis_name: `str` - Label of the Y axis of the plot. Default = `"Loss"`
        add_legend: `bool` - Boolean value indication whether a legend should be added to the plot. Default = `True`
        legend_loc: `str` - Location of the legend on the plot figure, if added. Default = `"upper right"`
    
        """
        self._ax.plot(self._train_losses, color=train_color, label=train_label)
        if self._val_losses:
            self._ax.plot(self._val_losses, color=val_color, label=val_label)
        self._ax.set_title(title)
        self._ax.set_xlabel(xaxis_name)
        self._ax.set_ylabel(yaxis_name)

        if add_legend:
            labels = [train_label]
            if self._val_losses:
                labels.append(val_label)
            self._ax.legend(labels, loc=legend_loc)

        return self._ax


class ROCPlotter:
    def __init__(
        self,
        ax: axes.Axes,
        fpr: np.ndarray,
        tpr: np.ndarray,
    ):
        self._ax = ax
        self._fpr = fpr
        self._tpr = tpr

    def plot(
        self,
        curve_color: str = "darkorange",
        title: str = "Receiver operating characteristic (ROC) curve",
        plot_label: str = "ROC Curve",
        xaxis_name: str = "False Positive Rate",
        yaxis_name: str = "True Positive Rate",
        add_legend: bool = True,
        legend_loc: str = "lower right",
    ) -> axes.Axes:
        self._ax.figure(figsize=(8, 6))
        self._ax.plot(self._fpr, self._tpr, color=curve_color, lw=2, label=plot_label)
        self._ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        self._ax.set_xlim([0.0, 1.0])
        self._ax.set_ylim([0.0, 1.05])
        self._ax.set_xlabel(xaxis_name)
        self._ax.set_ylabel(yaxis_name)
        self._ax.set_title(title)

        if add_legend:
            self._ax.legend(loc=legend_loc)

        return self._ax


class ConfusionMatrixPlotter:
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
        cmap: Colormap,
        title: str = "Confusion Matrix",
        xaxis_name: str = "True",
        yaxis_name: str = "Predicted",
        show_accuracy: bool = False,
    ) -> axes.Axes:
        self._ax.matshow(self._conf_matrix, cmap=cmap)
        self._ax.set_xlabel(xaxis_name)
        self._ax.set_ylabel(yaxis_name)
        self._ax.set_title(title)
        self._ax.set_xticks(
            np.arange(len(self._class_labels)), labels=self._class_labels
        )
        self._ax.tick_params(
            axis="x", bottom=True, top=False, labelbottom=True, labeltop=False
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

        if self._accuracy and show_accuracy:
            self._ax.annotate(
                text="Accuracy: %0.2f" % self._accuracy,
                xy=(40, 13),
                xycoords="figure pixels",
            )

        return self._ax

    def _get_text_color(self, row_idx, col_idx):
        max_val = self._conf_matrix.max()
        color = "black"
        if max_val > 0 and self._conf_matrix[row_idx, col_idx] / max_val > 0.5:
            color = "white"
        return color
