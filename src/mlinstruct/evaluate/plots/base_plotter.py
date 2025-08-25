from abc import ABC, abstractmethod
from matplotlib.colors import Colormap
from matplotlib import colormaps
import matplotlib.axes as axes

DEFAULT_CMAP: Colormap = colormaps.get_cmap("Blues")


class BasePlotter(ABC):
    """
    Base class for all plotters.

    Args:
        title (str): Title of the plot.
        xaxis_name (str): Name of the x-axis.
        yaxis_name (str): Name of the y-axis.
        cmap (Colormap, optional): Colormap to use for the plot. Defaults to DEFAULT_CMAP.
    """
    def __init__(self, title: str, xaxis_name: str, yaxis_name: str, cmap: Colormap = DEFAULT_CMAP):
        self.__title: str = title
        self.__xaxis_name: str = xaxis_name
        self.__yaxis_name: str = yaxis_name
        self.__cmap: Colormap = cmap

    @abstractmethod
    def plot(self, ax: axes.Axes, *args, **kwargs) -> axes.Axes:
        """
        Abstract plot method to be implemented by subclasses.
        Should accept an Axes object and plot-specific data.
        """
        pass
