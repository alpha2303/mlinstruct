from collections.abc import Iterable
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ._trainer import BaseTrainer
from .extensions import BaseExtension, ExtensionList
from ..utils import Result, Option, load_extensions

_DEFAULT_SAVE_PATH: Path = Path("./Models")
_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M"


class TorchTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        scheduler: Option[optim.lr_scheduler.LRScheduler] = Option.none(),
        save_folder_path: Path = _DEFAULT_SAVE_PATH,
        extensions: ExtensionList = ExtensionList(),
    ) -> None:
        self._model: nn.Module = model
        self._optimizer: optim.Optimizer = optimizer
        self._loss_fn = loss_fn
        self._scheduler: Option[optim.lr_scheduler.LRScheduler] = scheduler
        self._save_folder_path: Path = _DEFAULT_SAVE_PATH
        self._extensions: ExtensionList = extensions

    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> Result[tuple[np.ndarray, np.ndarray], Exception]:
        best_vloss = np.inf
        train_loss_list, test_loss_list = [], []
        self._training_timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)

        for epoch_index in range(0, n_iter):
            result = self._train_one_epoch(train_data)
            if result.is_err():
                return result
            avg_loss: float = result.unwrap()
            result = self._validate(test_data)
            if result.is_err():
                return result
            avg_vloss: float = result.unwrap()

            print(
                f"Epoch {epoch_index + 1}: Training Loss = {avg_loss} | Validation Loss = {avg_vloss} | Current LR = {self._optimizer.param_groups[0]['lr']}"
            )
            train_loss_list.append(avg_loss)
            test_loss_list.append(avg_vloss)

            if not self._scheduler.is_none():
                if isinstance(
                    self._scheduler.unwrap(), torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self._scheduler.unwrap().step(avg_vloss)
                else:
                    self._scheduler.unwrap().step()

            is_best: bool = False
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                is_best = True

            with self._extensions.get_extension("Checkpoint") as chk:
                if not chk.is_none():
                    result: Result = chk.unwrap().execute(
                        model=self._model,
                        is_best=is_best,
                        epoch=(epoch_index + 1),
                        optimizer=self._optimizer,
                        loss=avg_vloss,
                    )
                    if result.is_err():
                        return result

            with self._extensions.get_extension("EarlyStopper") as es:
                if not es.is_none():
                    result: Result = es.unwrap().execute(vloss=avg_vloss)
                    if result.is_err():
                        return result
                    print(f"Early stopping triggered at epoch: {epoch_index + 1}")
                    break

        return Result.ok((train_loss_list, test_loss_list))

    def _train_one_epoch(self, train_data: Iterable) -> Result[float, Exception]:
        running_loss: float = 0.0
        self._model.train()

        try:
            for _, data in enumerate(train_data):
                X_batch, Y_batch = data
                Y_pred = self._model(X_batch)
                loss = self._loss_fn(Y_pred, Y_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                running_loss += loss.item()
        except Exception as e:
            return Result.err(e)

        return running_loss / len(train_data)

    def _validate(self, test_data: Iterable) -> Result[float, Exception]:
        running_vloss: float = 0.0
        self._model.eval()

        try:
            with torch.no_grad():
                for _, vdata in enumerate(test_data):
                    vX_batch, vY_batch = vdata
                    vY_pred = self._model(vX_batch)
                    vloss = self._loss_fn(vY_pred, vY_batch)
                    running_vloss += vloss.item()
        except Exception as e:
            return Result.err(e)

        return running_vloss / len(test_data)
