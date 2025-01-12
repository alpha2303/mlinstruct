from collections.abc import Iterable
from typing import Optional
import numpy as np

from ..utils import EarlyStopper


class BaseTrainer:
    _early_stopper: Optional[EarlyStopper] = None

    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def config_early_stop(self, patience: int = 2, min_delta: float = 0.01) -> None:
        self._early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
