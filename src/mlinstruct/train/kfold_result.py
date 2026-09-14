from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlinstruct.train.train_result import TrainResult


@dataclass(frozen=True)
class KFoldResult:
    """The aggregate result of a K-Fold cross-validation training session.

    Args:
        model_name (str): The name of the model architecture trained across folds.
        model_save_path (pathlib.Path): The shared root directory containing each
            fold's checkpoint subdirectory.
        fold_results (list[TrainResult]): The per-fold training results, in fold order.
        mean_val_loss (float): The mean of each fold's best validation loss.
        std_val_loss (float): The standard deviation of each fold's best validation loss.
        best_fold_index (int): The 1-based index of the fold with the lowest best
            validation loss, matching the "fold_{i}" run-directory naming.
    """

    model_name: str
    model_save_path: Path
    fold_results: list[TrainResult]
    mean_val_loss: float
    std_val_loss: float
    best_fold_index: int

    @classmethod
    def from_fold_results(
        cls, model_name: str, model_save_path: Path, fold_results: list[TrainResult]
    ) -> "KFoldResult":
        """Build a KFoldResult by aggregating each fold's best validation loss.

        Args:
            model_name (str): The name of the model architecture trained across folds.
            model_save_path (pathlib.Path): The shared root directory containing each
                fold's checkpoint subdirectory.
            fold_results (list[TrainResult]): The per-fold training results, in fold order.

        Returns:
            KFoldResult: The aggregated result.
        """
        best_val_losses = [result.best_val_loss for result in fold_results]

        return cls(
            model_name=model_name,
            model_save_path=model_save_path,
            fold_results=fold_results,
            mean_val_loss=float(np.mean(best_val_losses)),
            std_val_loss=float(np.std(best_val_losses)),
            best_fold_index=int(np.argmin(best_val_losses)) + 1,
        )
