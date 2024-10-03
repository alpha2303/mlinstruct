from collections.abc import Iterable
from pathlib import Path
from typing import Self
import numpy as np

from ..utils import Result


class BaseTrainer:
    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> Result[tuple[np.ndarray, np.ndarray], Exception]:
        return Result.err(NotImplementedError())
    
    def load_model(self, model_file_path: Path, dirty_start: bool = False) -> Result[bool, Exception]:
        return Result.err(NotImplementedError())

    def _train_one_epoch(self, train_data: Iterable) -> Result[float, Exception]:
        return Result.err(NotImplementedError())

    def _validate(self, test_data: Iterable) -> Result[float, Exception]:
        return Result.err(NotImplementedError())
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool) -> Result[bool, Exception]:
        return Result.err(NotImplementedError())
