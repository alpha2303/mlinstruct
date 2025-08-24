import numpy as np
import matplotlib.axes as axes


class LossPlotter:
    """
    Creates a plotting object to visualize loss trends during training.
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
        train_label: str = "Train",
        val_color: str = "orange",
        val_label: str = "Validation",
        title: str = "Training Loss per Epoch",
        xaxis_name: str = "Epoch",
        yaxis_name: str = "Loss",
        add_legend: bool = True,
        legend_loc: str = "upper right",
    ) -> axes.Axes:
        """
        plot() -> Generates the Training Loss Line Plot on `matplotlib.axes.Axes` object provided.
        Arguments follow the options provided by Matplotlib.

        Validation Loss will be plotted only if validation loss values were provided at object creation.

        Arguments:
        train_color: `str` - Color of the training loss plot. Default = `"orange"`
        train_label: `str` - Label of the training loss plot. Default = `"Train"`
        val_color: `str` - Color of the validation loss plot. Default = `"orange"`
        val_label: `str` - Label of the validation loss plot. Default = `"Validation"`
        title: `str` - Title of the plot. Default = `"Training Loss"`
        xaxis_name: `str` - Label of the X axis of the plot. Default = `"Epoch"`
        yaxis_name: `str` - Label of the Y axis of the plot. Default = `"Loss"`
        add_legend: `bool` - Boolean value indication whether a legend should be added to the plot. Default = `True`
        legend_loc: `str` - Location of the legend on the plot figure, if added. Default = `"upper right"`

        """
        self._ax.plot(self._train_losses, color=train_color, label=train_label)
        if self._val_losses.any():
            self._ax.plot(self._val_losses, color=val_color, label=val_label)
        self._ax.set_title(title)
        self._ax.set_xlabel(xaxis_name)
        self._ax.set_ylabel(yaxis_name)

        if add_legend:
            labels = [train_label]
            if self._val_losses.any():
                labels.append(val_label)
            self._ax.legend(labels, loc=legend_loc)

        return self._ax
