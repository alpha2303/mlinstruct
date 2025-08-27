from abc import ABC, abstractmethod
from typing import Optional, Self

from matplotlib.axes import Axes
import numpy as np


class BaseMetrics(ABC):
    @classmethod
    @abstractmethod
    def from_predictions(cls, y: np.ndarray, y_pred: np.ndarray, *args, **kwargs) -> Self:
        """
        Create an instance of the metric from the true and predicted values.
        """
        pass

    @abstractmethod
    def plot(
        self,
        title: str,
        xaxis_name: str,
        yaxis_name: str,
        ax: Optional[Axes] = None,
        *args,
        **kwargs,
    ) -> Axes:
        """
        Plot the metric on the given matplotlib.axes.Axes object.
        """
        pass
