from pathlib import Path
from typing import Iterable


class BaseModelProxy:
    def load_weights(self, model_file_path: Path) -> None:
        raise NotImplementedError()

    def save_weights(
        self, epoch: int, loss: float, save_folder_path: Path, model_name: str
    ) -> None:
        raise NotImplementedError()

    def get_lr(self) -> float:
        raise NotImplementedError()

    def has_scheduler(self) -> bool:
        raise NotImplementedError()

    def step(self, avg_vloss: float) -> None:
        raise NotImplementedError()

    def train_one_epoch(self, trainLoader: Iterable) -> float:
        raise NotImplementedError()

    def validate(self, test_data: Iterable) -> float:
        raise NotImplementedError()
    
    def summary(self) -> None:
        raise NotImplementedError()
