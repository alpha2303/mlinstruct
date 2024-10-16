from pathlib import Path
from typing import Iterable

import torch
import torchinfo

from ...utils import Option
from ._base_model_proxy import BaseModelProxy


class TorchModelProxy(BaseModelProxy):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss,
    ):
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._scheduler: Option[torch.optim.lr_scheduler.LRScheduler] = Option.none()

    def load_weights(self, model_file_path: Path) -> None:
        try:
            checkpoint = torch.load(model_file_path)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            raise e

        self._model.eval()

        return True

    def save_weights(
        self, epoch: int, loss: float, save_folder_path: Path, model_name: str
    ) -> None:
        if not save_folder_path.exists():
            raise Exception(
                "Model save path does not exist. If you are running the save method directly, ensure that the save path is valid."
            )

        try:
            model_object = {
                "epoch": epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": loss,
            }

            torch.save(model_object, save_folder_path.joinpath(f"{model_name}.pt"))
        except Exception as e:
            raise e

    def set_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        self._scheduler = Option.some(scheduler)

    def get_lr(self) -> float:
        return self._optimizer.param_groups[0]["lr"]

    def has_scheduler(self) -> None:
        return not self._scheduler.is_none()

    def step(self, avg_vloss: float) -> None:
        if self.has_scheduler():
            if isinstance(
                self._scheduler.unwrap(), torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self._scheduler.unwrap().step(avg_vloss)
            else:
                self._scheduler.unwrap().step()

    def train_one_epoch(self, trainLoader: Iterable) -> float:
        running_loss = 0.0
        self._model.train()

        for _, data in enumerate(trainLoader):
            X_batch, Y_batch = data
            Y_pred = self._model(X_batch)
            loss = self._loss_fn(Y_pred, Y_batch)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            running_loss += loss.item()

        return running_loss / len(trainLoader)

    def validate(self, test_data: Iterable) -> float:
        running_vloss: float = 0.0
        self._model.eval()

        try:
            with torch.no_grad():
                for _, vdata in enumerate(test_data):
                    vX_batch, vY_batch = vdata
                    vY_pred: torch.Tensor = self._model(vX_batch)
                    vloss = self._loss_fn(vY_pred, vY_batch)
                    running_vloss += vloss.item()
        except Exception as e:
            raise e

        return running_vloss / len(test_data)
    
    def summary(self) -> None:
        print(torchinfo.summary(self._model))