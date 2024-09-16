from collections.abc import Iterable
import numpy as np

from .extensions import BaseExtension

from ..utils import Result


class BaseTrainer:
    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> Result[tuple[np.ndarray, np.ndarray], Exception]:
        return Result.err(NotImplementedError())

    def _train_one_epoch(self, train_data: Iterable) -> Result[float, Exception]:
        return Result.err(NotImplementedError())

    def _validate(self, test_data: Iterable) -> Result[float, Exception]:
        return Result.err(NotImplementedError())
