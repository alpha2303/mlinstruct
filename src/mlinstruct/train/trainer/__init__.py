from typing import Any

from mlinstruct.train.trainer.base_trainer import BaseTrainer
from mlinstruct.train.trainer.default_trainer import DefaultTrainer
from mlinstruct.train.trainer.kfold_trainer import KFoldTrainer
from mlinstruct.utils.optional_deps import require

__all__ = ["BaseTrainer", "DefaultTrainer", "KFoldTrainer", "GANTrainer"]


def __getattr__(name: str) -> Any:
    if name == "GANTrainer":
        require("torch", extra="torch", symbol="GANTrainer")
        from mlinstruct.train.trainer.torch_gan_trainer import GANTrainer

        return GANTrainer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
