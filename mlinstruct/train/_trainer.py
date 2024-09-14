from collections.abc import Iterable
import numpy as np

from mlinstruct.train.extensions import BaseExtension
from mlinstruct.utils import Result

class BaseTrainer:
    def train(self, train_data: Iterable, test_data: Iterable, n_iter: int) -> Result[tuple[object]]:
        raise NotImplementedError()

    def _epoch_loop(self, train_data: Iterable) -> Result[float]:
        raise NotImplementedError()
    
    def _validate(self, test_data: Iterable) -> Result[float]:
        raise NotImplementedError()
