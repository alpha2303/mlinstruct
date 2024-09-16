from datetime import datetime
from pathlib import Path
from typing import Any
import torch

from ._base_extension import BaseExtension
from ...utils import Result, Option, check_params


class BaseCheckpoint(BaseExtension):
    def __init__(self, save_folder_path: Path, file_ext: str) -> None:
        self._save_folder_path: Path = save_folder_path
        self._file_ext: str = file_ext
        self.__init__(key="Checkpoint")

    def execute(self, **kwargs) -> Result[bool, Exception]:
        result: Result[bool, Exception] = check_params(kwargs, {"model": Any})
        if result.is_err():
            return result

        if "is_best" in kwargs and kwargs.get("is_best"):
            result = self.save(model=kwargs.get("model"), name="best" + self._file_ext)
            if result.is_err():
                return result

        return self.save(kwargs.get("model"), name="last" + self._file_ext)

    def save(self, **kwargs) -> Result[bool, Exception]:
        return Result.err(NotImplemented())


class TorchCheckpoint(BaseCheckpoint):
    def __init__(self, save_folder_path: Path) -> None:
        super().__init__(save_folder_path)

    def save(self, **kwargs) -> Result[bool, Exception]:
        result: Result[bool, Exception] = check_params(
            kwargs,
            {
                "model": torch.nn.Module,
                "name": str,
                "epoch": float,
                "optimizer": torch.optim.Optimizer,
                "loss": float,
            },
        )
        if result.is_err():
            return result

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
            return Result.ok(True)
        except Exception as e:
            return Result.err(e)
