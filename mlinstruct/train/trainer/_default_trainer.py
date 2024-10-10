from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from ...utils import Option
from ..model_proxy._base_model_proxy import BaseModelProxy
from ._base_trainer import BaseTrainer
from ..utils import EarlyStopper

_DEFAULT_SAVE_PATH: Path = Path("./Models")
_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M"


class DefaultTrainer(BaseTrainer):
    def __init__(
        self,
        model_proxy: BaseModelProxy,
        save_folder_path: Path = _DEFAULT_SAVE_PATH,
        save_checkpoint: bool = True,
    ) -> None:
        self._model_proxy = model_proxy
        self._save_folder_path = save_folder_path
        self._save_checkpoint: bool = save_checkpoint
        self._early_stopper: Option[EarlyStopper] = Option.none()

    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._model_proxy.get_start_epoch() > n_iter:
            raise ValueError(
                f"Start Epoch ({self._model_proxy.get_start_epoch()}) is greater than iteration count (n_iter = {n_iter}). Disable dirty start if you are loading an existing model with a greater trained epoch."
            )

        best_vloss: float = np.inf
        train_loss_list, test_loss_list = [], []
        model_save_path = self._save_folder_path.joinpath(
            datetime.now().strftime(_TIMESTAMP_FORMAT)
        )

        for epoch_index in range(self._model_proxy.get_start_epoch(), n_iter):
            avg_loss = self._model_proxy.train_one_epoch(train_data)

            avg_vloss = self._model_proxy.validate(test_data)

            print(
                f"Epoch {epoch_index + 1}: Training Loss = {avg_loss} | Validation Loss = {avg_vloss} | Current LR = {self._model_proxy.get_lr()}"
            )
            train_loss_list.append(avg_loss)
            test_loss_list.append(avg_vloss)

            if self._model_proxy.has_scheduler():
                self._model_proxy.step(avg_vloss=avg_loss)

            is_best: bool = False
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                is_best = True

            if self._save_checkpoint:
                if not model_save_path.exists():
                    model_save_path.mkdir(parents=True)

                if is_best:
                    self._model_proxy.save_weights(
                        epoch_index + 1, avg_vloss, model_save_path, "best"
                    )

                self._model_proxy.save_weights(
                    epoch_index + 1, avg_vloss, model_save_path, "last"
                )

            if (
                not self._early_stopper.is_none()
                and self._early_stopper.unwrap().early_stop(avg_vloss)
            ):
                print(f"Early stop triggered at epoch: {epoch_index + 1}")
                break

        self._model_proxy.set_start_epoch(0)
        if self._save_checkpoint:
            print(f"Model saved to {model_save_path.resolve()}")

        return (train_loss_list, test_loss_list)

    def allow_early_stop(self, patience: float = 5, min_delta: float = 0.0):
        self._early_stopper = Option.some(
            EarlyStopper(patience=patience, min_delta=min_delta)
        )
