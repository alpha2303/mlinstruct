from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Self
import numpy as np

from ..model_proxy._base_model_proxy import BaseModelProxy
from ..utils import EarlyStopper, CheckpointWriter

_DEFAULT_SAVE_PATH: Path = Path("./Models")


class BaseTrainer:
    _model_proxy: BaseModelProxy
    _early_stopper: Optional[EarlyStopper] = None
    _checkpoint_writer: Optional[CheckpointWriter] = None

    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    def add_model_proxy(self, model_proxy: BaseModelProxy) -> Self:
        self._model_proxy = model_proxy
        return self

    def add_early_stop(self, patience: int = 2, min_delta: float = 0.01) -> Self:
        self._early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        return self
    
    def add_checkpoint_save(self, save_root_dirpath: Path = _DEFAULT_SAVE_PATH) -> Self:
        self._checkpoint_writer = CheckpointWriter(save_root_dirpath)
        return self
