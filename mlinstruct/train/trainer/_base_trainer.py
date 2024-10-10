from collections.abc import Iterable
import numpy as np


class BaseTrainer:
    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def allow_early_stop(self, patience: int, min_delta: float) -> None:
        raise NotImplementedError()
