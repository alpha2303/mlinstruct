from datetime import datetime
from pathlib import Path

from mlinstruct.train.model_proxy import BaseModelProxy

_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"


class CheckpointWriter:
    """
    Class to write model checkpoints to disk.

    Args:
        root_save_dir_path (Path): The folder path where run directories will be created.
        run_name (Optional[str]): Name for the run directory. Defaults to a
            timestamp. A name that already exists under root_save_dir_path
            gets a numeric suffix (_1, _2, ...) instead of being reused.
    """

    _root_save_dir_path: Path
    _run_name: str | None
    _model_save_dir_path: Path

    def __init__(self, root_save_dir_path: Path, run_name: str | None = None) -> None:
        self._root_save_dir_path = root_save_dir_path
        self._run_name = run_name

    def regenerate_model_save_path(self) -> None:
        """Regenerate the model save path, avoiding collisions with existing run directories."""
        base_name = self._run_name or datetime.now().strftime(_TIMESTAMP_FORMAT)

        candidate: Path = self._root_save_dir_path.joinpath(base_name)
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = self._root_save_dir_path.joinpath(f"{base_name}_{suffix}")

        self._model_save_dir_path = candidate
        self._model_save_dir_path.mkdir(parents=True)

    def get_model_save_path(self) -> Path:
        """Get the model save path.

        Returns:
            pathlib.Path: The model save path.
        """
        return self._model_save_dir_path

    def create_checkpoint(self, model_proxy: BaseModelProxy, epoch: int, vloss: float) -> Path:
        """Create a model checkpoint.

        Args:
            model_proxy (BaseModelProxy): The model proxy to use for saving weights.
            epoch (int): The current epoch number.
            vloss (float): The validation loss for the current epoch.

        Returns:
            Path: The path the checkpoint was written to.
        """
        model_stem: str = f"model_epoch_{epoch}_vloss_{vloss:.4f}"
        return model_proxy.save_weights(epoch, self._model_save_dir_path, model_stem, loss=vloss)
