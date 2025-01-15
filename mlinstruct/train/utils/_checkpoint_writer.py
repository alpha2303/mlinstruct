from datetime import datetime
from pathlib import Path
from ..model_proxy import BaseModelProxy


_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M"


class CheckpointWriter:
    def __init__(self, save_folder_path: Path) -> None:
        self._save_folder_path: Path = save_folder_path
        self._model_save_path: Path = self.regenerate_path()

    def regenerate_path(self) -> None:
        self._model_save_path = self._save_folder_path.joinpath(
            datetime.now().strftime(_TIMESTAMP_FORMAT)
        )
        if not self._model_save_path.exists():
            self._model_save_path.mkdir(parents=True)

    def get_model_save_path(self) -> Path:
        return self._model_save_path

    def create_checkpoint(
        self, model_proxy: BaseModelProxy, epoch: int, vloss: float, is_best: bool
    ) -> None:
        if is_best:
            model_proxy.save_weights(epoch, vloss, self._model_save_path, "best")

        model_proxy.save_weights(epoch + 1, vloss, self._model_save_path, "last")
