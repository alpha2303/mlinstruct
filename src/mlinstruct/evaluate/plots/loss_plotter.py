from typing import Optional
import numpy as np
import matplotlib.axes as axes

from .base_plotter import BasePlotter


class LossPlotter(BasePlotter):
    """
    Creates a plotting object to visualize loss trends during training.
    Validation loss information can also be included for comparison.

    Args:
        train_color (str, optional): Color of the training loss plot. Defaults to "blue".
        train_label (str, optional): Label of the training loss plot. Defaults to "Train".
        val_color (str, optional): Color of the validation loss plot. Defaults to "orange".
        val_label (str, optional): Label of the validation loss plot. Defaults to "Validation".
        title (str, optional): Title of the plot. Defaults to "Training Loss per Epoch".
        xaxis_name (str, optional): Label of the X axis of the plot. Defaults to "Epoch".
        yaxis_name (str, optional): Label of the Y axis of the plot. Defaults to "Loss".
        add_legend (bool, optional): Whether a legend should be added to the plot. Defaults to True.
        legend_loc (str, optional): Location of the legend on the plot figure. Defaults to "upper right".
        cmap (Colormap, optional): Colormap to use for the plot.
            Defaults to `matplotlib.pyplot.cm.Blues`.
    """

    def __init__(
        self,
        title: str = "Training Loss per Epoch",
        xaxis_name: str = "Epoch",
        yaxis_name: str = "Loss",
        train_color: str = "blue",
        train_label: str = "Train",
        val_color: str = "orange",
        val_label: str = "Validation",
        add_legend: bool = True,
        legend_loc: str = "upper right",
    ):
        self.__title: str = title
        self.__xaxis_name: str = xaxis_name
        self.__yaxis_name: str = yaxis_name
        self.__train_color: str = train_color
        self.__train_label: str = train_label
        self.__val_color: str = val_color
        self.__val_label: str = val_label
        self.__add_legend: bool = add_legend
        self.__legend_loc: str = legend_loc

    def plot(
        self,
        ax: axes.Axes,
        train_losses: np.ndarray,
        val_losses: Optional[np.ndarray] = None,
        **kwargs,
    ) -> axes.Axes:
        """Generates the Training Loss Line Plot on matplotlib.axes.Axes object provided.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib Axes object on which the plot will be drawn.
            train_losses (numpy.ndarray): NumPy array containing the training loss values of each training epoch.
            val_losses (numpy.ndarray, optional): NumPy array containing the validation loss values of each training epoch.
                Defaults to None.

        Returns:
            matplotlib.axes.Axes: The axes object with the plotted data.
        """
        ax.plot(train_losses, color=self.__train_color, label=self.__train_label)
        if val_losses is not None:
            ax.plot(val_losses, color=self.__val_color, label=self.__val_label)
        ax.set_title(self.__title)
        ax.set_xlabel(self.__xaxis_name)
        ax.set_ylabel(self.__yaxis_name)

        if self.__add_legend:
            labels = [self.__train_label]
            if val_losses is not None:
                labels.append(self.__val_label)
            ax.legend(labels, loc=self.__legend_loc)

        return ax
