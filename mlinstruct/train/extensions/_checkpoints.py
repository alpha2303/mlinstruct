from datetime import datetime
from pathlib import Path
import torch

from mlinstruct.train.extensions import BaseExtension
from mlinstruct.utils import Result, Option


class BaseCheckpoint(BaseExtension):
    def __init__(self, save_folder_path: Path, file_ext: str) -> None:
        self._save_folder_path: Path = save_folder_path
        self._file_ext: str = file_ext
        self.__init__(key="Checkpoint")

    def execute(self, **kwargs) -> Result[bool, Exception]:
        if "model" not in kwargs:
            return Result.err(ValueError("The parameter 'model' is not provided."))

        if "is_best" in kwargs and kwargs.get("is_best"):
            save_result: Result[bool, Exception] = self.save(kwargs.get("model"), name="best"+self._file_ext)
            if save_result.is_err():
                return save_result

        return self.save(kwargs.get("model"), name="last"+self._file_ext)

    def save(self, model: object, name: str, **kwargs) -> Result[bool, Exception]:
        return Result.err(NotImplemented())


class TorchCheckpoint(BaseCheckpoint):
    def __init__(self, save_folder_path: Path) -> None:
        super().__init__(save_folder_path)
    
    def save(self, model: object, name: str, **kwargs) -> Result[bool, Exception]:
        full_save_path: Path = self._save_folder_path.joinpath(name)
        if not isinstance(model, torch.nn.Module):
            return Result.err(ValueError(f"The parameter 'model: {type(model)}' is not an instance of torch.nn.Module class."))
        build_result: Result[dict, Exception] = self._build_save_object(model, kwargs)
        if build_result.is_err():
            return build_result
        try:
            torch.save(build_result.unwrap(), full_save_path)
        except Exception as e:
            return Result.err(e)
    
    def _build_save_object(model: torch.nn.Module, kwargs: dict) -> Result[dict, Exception]:
        # if "epoch" not in kwargs:
        #     return Result.err(ValueError("The parameter 'epoch: float' is not provided."))
        # return {
        #     "epoch": epoch,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "loss": loss
        # }
        pass

