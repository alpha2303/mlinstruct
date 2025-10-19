from pathlib import Path
from typing import List


class TrainResult:
    """Class representing the result of a training session.

    Args:
        model_name (str): The name of the model.
        model_save_path (pathlib.Path): The path to save the model.
        epochs (int): The number of epochs trained.
        train_loss_list (List[float]): The list of training losses.
        val_loss_list (List[float]): The list of validation losses.
    """

    model_name: str
    model_save_path: Path
    epochs: int
    train_loss_list: List[float]
    val_loss_list: List[float]

    def __init__(
        self,
        model_name: str,
        model_save_path: Path,
        epochs: int,
        train_loss_list: List[float],
        val_loss_list: List[float],
    ) -> None:
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.train_loss_list = train_loss_list
        self.val_loss_list = val_loss_list
