from datetime import datetime
from pathlib import Path
from ..model_proxy import BaseModelProxy


_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M"


class CheckpointWriter:
    """
    Class to write model checkpoints to disk.

    Args:
        save_folder_path (Path): The folder path where checkpoints will be saved.
    """

    __root_save_dir_path: Path
    __model_save_dir_path: Path

    def __init__(self, root_save_dir_path: Path) -> None:
        self.__root_save_dir_path = root_save_dir_path
        self.regenerate_model_save_path()

    def regenerate_model_save_path(self) -> None:
        """Regenerate the model save path based on the current timestamp."""
        self.__model_save_dir_path = self.__root_save_dir_path.joinpath(
            datetime.now().strftime(_TIMESTAMP_FORMAT)
        )
        if not self.__model_save_dir_path.exists():
            self.__model_save_dir_path.mkdir(parents=True)

    def get_model_save_path(self) -> Path:
        """Get the model save path.

        Returns:
            pathlib.Path: The model save path.
        """
        return self.__model_save_dir_path

    def create_checkpoint(
        self, model_proxy: BaseModelProxy, epoch: int, vloss: float
    ) -> None:
        """Create a model checkpoint.

        Args:
            model_proxy (BaseModelProxy): The model proxy to use for saving weights.
            epoch (int): The current epoch number.
            vloss (float): The validation loss for the current epoch.
        """
        model_name: str = f"model_epoch_{epoch}_vloss_{vloss:.4f}.pt"
        model_proxy.save_weights(
            epoch, self.__model_save_dir_path, model_name, loss=vloss
        )
