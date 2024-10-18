
from typing import Self
from matplotlib.axes import Axes
import numpy as np


class BaseMetric():
    @classmethod
    def from_predictions(cls, y: np.ndarray, y_pred: np.ndarray) -> Self:
        raise NotImplementedError()
    
    def plot(self) -> Axes:
        raise NotImplementedError()