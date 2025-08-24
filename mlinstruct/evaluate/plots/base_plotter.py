from abc import ABC, abstractmethod
from matplotlib.colors import Colormap
from matplotlib.pyplot import axes, cm

__DEFAULT_CMAP: Colormap = cm.Blues


class BasePlotter(ABC):
    """
    Base class for all plotters.

    Args:
        title (str): Title of the plot.
        xaxis_name (str): Name of the x-axis.
        yaxis_name (str): Name of the y-axis.
        cmap (Colormap, optional): Colormap to use for the plot. Defaults to __DEFAULT_CMAP.
    """
    def __init__(self, title: str, xaxis_name: str, yaxis_name: str, cmap: Colormap = __DEFAULT_CMAP):
        self.title: str = title
        self.xaxis_name: str = xaxis_name
        self.yaxis_name: str = yaxis_name
        self.cmap: Colormap = cmap

    @abstractmethod
    def plot(self, ax: axes.Axes, *args, **kwargs) -> axes.Axes:
        """
        Abstract plot method to be implemented by subclasses.
        Should accept an Axes object and plot-specific data.
        """
        pass
