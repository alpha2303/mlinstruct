from collections.abc import Iterable
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

from ._base_trainer import BaseTrainer
from ._early_stopper import EarlyStopper
from ..utils import Result, Option

_DEFAULT_SAVE_PATH: Path = Path("./Models")
_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M"


class TorchTrainer(BaseTrainer):
    """
    Creates a trainer that abstracts the model training loop for PyTorch models (nn.Module).

    Required Arguments:
    - *model*: `torch.nn.Module` - A PyTorch model object that inherits the `torch.nn.Module` class.
    - *loss_fn*: `torch.nn.modules.loss._Loss` - A PyTorch loss function object that returns an output that supports `backward()` call.
    - *optimizer*: `torch.optim.Optimizer` - Optimizer object that inherits the `torch.optim.Optimizer` class.

    Additional Arguments:
    - *save_folder_path*: `pathlib.Path` - A `pathlib.Path` object pointing to the file location for model checkpoint storage. Default: `./Models`
    - *checkpoint*: `bool` - `bool` value to enable saving of model checkpoints after each training epoch. Default: `False`
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        save_folder_path: Path = _DEFAULT_SAVE_PATH,
        checkpoint: bool = False,
    ) -> None:
        self._model: torch.nn.Module = model
        self._optimizer: torch.optim.Optimizer = optimizer
        self._loss_fn = loss_fn
        self._save_folder_path: Path = save_folder_path
        self._checkpoint: bool = checkpoint

        self._start_epoch: int = 0
        self._model_save_path: Option[Path] = Option.none()
        self._scheduler: Option[torch.optim.lr_scheduler.LRScheduler] = Option.none()
        self._early_stopper: Option[EarlyStopper] = Option.none()

    def train(
        self, train_data: Iterable, test_data: Iterable, n_iter: int
    ) -> Result[tuple[np.ndarray, np.ndarray], Exception]:
        if self._start_epoch > n_iter:
            return Result.err(
                ValueError(
                    f"Start Epoch ({self._start_epoch}) is greater than iteration count (n_iter = {n_iter}). Disable dirty start if you are loading an existing model with a greater trained epoch."
                )
            )

        best_vloss: float = np.inf
        train_loss_list, test_loss_list = [], []
        self._model_save_path = Option.some(
            self._save_folder_path.joinpath(datetime.now().strftime(_TIMESTAMP_FORMAT))
        )

        for epoch_index in range(self._start_epoch, n_iter):
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

            if self._checkpoint:
                result: Result[bool, Exception] = self._save_checkpoint(
                    epoch_index + 1, avg_vloss, is_best
                )
                if result.is_err():
                    return result

            if not self._early_stopper.is_none():
                result: Result[bool, Exception] = (
                    self._early_stopper.unwrap().early_stop(avg_vloss)
                )
                if result.is_err():
                    return result

                if result.unwrap():
                    print(f"Early stop triggered at epoch: {epoch_index + 1}")
                    break

        return Result.ok((train_loss_list, test_loss_list))

    def load_model(
        self, model_file_path: Path, dirty_start: bool = False
    ) -> Result[bool, Exception]:
        try:
            checkpoint = torch.load(model_file_path)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._start_epoch = checkpoint["epoch"] if dirty_start else 0
        except Exception as e:
            return Result.err(e)

        self._model.eval()

        return True

    def add_lr_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        """
        Add a PyTorch learning rate scheduler to the Trainer

        Required Arguments:
        - *scheduler*: `torch.optim.lr_scheduler.LRScheduler` - A learning rate scheduler object that inherits the `torch.optim.lr_scheduler.LRScheduler` class.
        """

        self._scheduler = Option.some(scheduler)
        return True

    def allow_early_stop(self, patience: int = 2, min_delta: float = 0.0) -> None:
        """
        Allows the trainer to create an `mlinstruct.train.EarlyStopper` object, which enables early stop of model training.

        Default Arguments:
        - *patience*: `int` - The maximum count of epochs with consecutively decreasing loss, after which training stops. Default = `2`
        - *min_delta*: `float` - Minimum difference between current and previous loss values required for comparison (Maximum tolerable difference). Default = `0.0`
        """

        self._early_stopper = Option.some(
            EarlyStopper(patience=patience, min_delta=min_delta)
        )

    def _train_one_epoch(self, train_data: Iterable) -> Result[float, Exception]:
        running_loss: float = 0.0
        self._model.train()

        try:
            for _, data in enumerate(train_data):
                X_batch, Y_batch = data
                Y_pred: torch.Tensor = self._model(X_batch)
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
                    vY_pred: torch.Tensor = self._model(vX_batch)
                    vloss = self._loss_fn(vY_pred, vY_batch)
                    running_vloss += vloss.item()
        except Exception as e:
            return Result.err(e)

        return running_vloss / len(test_data)

    def _save_checkpoint(
        self, epoch: int, loss: float, is_best: bool
    ) -> Result[bool, Exception]:
        if self._model_save_path.is_none():
            return Result.err(
                Exception(
                    "Model save path has not been set yet. Do not call this method directly."
                )
            )

        model_object = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "loss": loss,
        }

        try:
            if is_best:
                torch.save(
                    model_object, self._model_save_path.unwrap().joinpath("best.pt")
                )
            torch.save(model_object, self._model_save_path.unwrap().joinpath("last.pt"))
        except Exception as e:
            return Result.err(e)
