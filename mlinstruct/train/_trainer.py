from collections.abc import Iterable
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mlinstruct.train.extensions import BaseExtension
from mlinstruct.utils import Result, Option

_DEFAULT_SAVE_PATH: Path = Path("./Models")
_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M"

class BaseTrainer:
    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> Result[tuple[np.ndarray, np.ndarray], Exception]:
        return Result.err(NotImplementedError())

    def _train_one_epoch(self, train_data: Iterable) -> Result[float, Exception]:
        return Result.err(NotImplementedError())

    def _validate(self, test_data: Iterable) -> Result[float, Exception]:
        return Result.err(NotImplementedError())


class TorchTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        
        scheduler: Option[optim.lr_scheduler.LRScheduler] = Option.none(),
        save_folder_path: Path = _DEFAULT_SAVE_PATH,
    ) -> None:
        self._model: nn.Module = model
        self._optimizer: optim.Optimizer = optimizer
        self._scheduler: Option[optim.lr_scheduler.LRScheduler] = scheduler
        self._save_folder_path: Path = _DEFAULT_SAVE_PATH
    
    def train(self, train_data: Iterable, test_data: Iterable, n_iter: int) -> Result[tuple[np.ndarray, np.ndarray], Exception]:
        return super().train(train_data, test_data, n_iter)

    def _train_one_epoch(self, train_data: Iterable) -> Result[float, Exception]:
        running_loss = 0.0
        last_loss = 0.0
        self._model.train()

        for _, data in enumerate(train_data):
            X_batch, Y_batch = data
            Y_pred = self._model(X_batch)
            loss = self._loss_fn(Y_pred, Y_batch)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_data)

    def _validate(self, test_data: Iterable) -> Result[float, Exception]:
        running_vloss = 0.
        self._model.eval()
        
        with torch.no_grad():
            for _, vdata in enumerate(test_data):
                vX_batch, vY_batch = vdata
                vY_pred = self._model(vX_batch)
                vloss = self._loss_fn(vY_pred, vY_batch)
                running_vloss += vloss.item()

        return running_vloss / len(test_data)
