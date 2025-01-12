from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..model_proxy._base_model_proxy import BaseModelProxy
from ._base_trainer import BaseTrainer
from ..utils import CheckpointWriter

_DEFAULT_SAVE_PATH: Path = Path("./Models")


class DefaultTrainer(BaseTrainer):
    def __init__(
        self,
        model_proxy: BaseModelProxy,
        save_root_dir: Optional[Path] = _DEFAULT_SAVE_PATH,
    ) -> None:
        self._model_proxy = model_proxy
        self._checkpoint_writer: Optional[CheckpointWriter] = None
        if save_root_dir:
            self._checkpoint_writer: CheckpointWriter = CheckpointWriter(save_root_dir)

    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        best_vloss: float = np.inf
        train_loss_list, test_loss_list = [], []
        if self._checkpoint_writer:
            self._checkpoint_writer.regenerate_path()

        for epoch_index in range(0, n_iter):
            avg_loss = self._model_proxy.train_one_epoch(train_data)

            avg_vloss = self._model_proxy.validate(test_data)

            print(
                f"Epoch {epoch_index + 1}: Training Loss = {avg_loss} | Validation Loss = {avg_vloss} | Learning Rate = {self._model_proxy.get_lr()}"
            )

            train_loss_list.append(avg_loss)
            test_loss_list.append(avg_vloss)

            if self._model_proxy.has_scheduler():
                self._model_proxy.step(avg_vloss=avg_loss)

            is_best: bool = False
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                is_best = True

            if self._checkpoint_writer:
                self._checkpoint_writer.create_checkpoint(
                    self._model_proxy, epoch_index + 1, avg_vloss, is_best
                )

            if self._early_stopper and self._early_stopper.early_stop(avg_vloss):
                print(f"Early stop triggered at epoch: {epoch_index + 1}")
                break

        if self._checkpoint_writer:
            print(
                f"Model checkpoints saved to {self._checkpoint_writer.get_model_save_path().resolve()}"
            )

        return (train_loss_list, test_loss_list)
