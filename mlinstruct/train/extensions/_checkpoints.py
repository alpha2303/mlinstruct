from datetime import datetime
from pathlib import Path
from typing import Any
import torch

from mlinstruct.train.extensions import BaseExtension
from mlinstruct.utils import Result, Option, check_params


class BaseCheckpoint(BaseExtension):
    def __init__(self, save_folder_path: Path, file_ext: str) -> None:
        self._save_folder_path: Path = save_folder_path
        self._file_ext: str = file_ext
        self.__init__(key="Checkpoint")

    def execute(self, **kwargs) -> Result[bool, Exception]:
        check_result: Result[bool, Exception] = check_params(kwargs, {"model": Any})
        if check_result.is_err():
            return check_result

        if "is_best" in kwargs and kwargs.get("is_best"):
            save_result: Result[bool, Exception] = self.save(
                model=kwargs.get("model"), name="best" + self._file_ext
            )
            if save_result.is_err():
                return save_result

        return self.save(kwargs.get("model"), name="last" + self._file_ext)

    def save(self, **kwargs) -> Result[bool, Exception]:
        return Result.err(NotImplemented())


class TorchCheckpoint(BaseCheckpoint):
    def __init__(self, save_folder_path: Path) -> None:
        super().__init__(save_folder_path)

    def save(self, **kwargs) -> Result[bool, Exception]:
        check_result = check_params(
            kwargs,
            {
                "model": torch.nn.Module,
                "name": str,
                "epoch": float,
                "optimizer": torch.optim.Optimizer,
                "loss": float,
            },
        )
        if check_result.is_err():
            return check_result

        try:
            torch.save(
                {
                    "epoch": kwargs.get("epoch"),
                    "model_state_dict": kwargs.get("model").state_dict(),
                    "optimizer_state_dict": kwargs.get("optimizer").state_dict(),
                    "loss": kwargs.get("loss"),
                },
                self._save_folder_path.joinpath(kwargs.get("name")),
            )
        except Exception as e:
            return Result.err(e)
