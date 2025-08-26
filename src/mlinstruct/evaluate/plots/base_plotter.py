from abc import ABC, abstractmethod
import matplotlib.axes as axes


class BasePlotter(ABC):
    """
    Base class for all plotters.
    """

    @abstractmethod
    def plot(self, ax: axes.Axes, *args, **kwargs) -> axes.Axes:
        """
        Abstract plot method to be implemented by subclasses.
        Should accept an Axes object and plot-specific data.
        """
        pass
