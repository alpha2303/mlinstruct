import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold

from mlinstruct.train.data_payload.base_data_payload import BaseDataPayload
from mlinstruct.train.kfold_result import KFoldResult
from mlinstruct.train.model_proxy.base_model_proxy import BaseModelProxy
from mlinstruct.train.trainer.base_trainer import DEFAULT_SAVE_PATH, BaseTrainer
from mlinstruct.train.trainer.default_trainer import DefaultTrainer
from mlinstruct.train.utils.checkpoint_writer import CheckpointWriter
from mlinstruct.utils.exception import TrainerError

TrainerFactory = Callable[[BaseModelProxy, BaseDataPayload, Path, str], BaseTrainer]


def _default_trainer_factory(
    model_proxy: BaseModelProxy,
    data_payload: BaseDataPayload,
    save_dir_path: Path,
    run_name: str,
) -> BaseTrainer:
    return DefaultTrainer(
        model_proxy=model_proxy,
        data_payload=data_payload,
        save_dir_path=save_dir_path,
        run_name=run_name,
    )


class KFoldTrainer:
    """Orchestrates K-Fold cross-validation training.

    Trains one independent model per cross-validation fold via a caller-supplied
    model_proxy_factory and a delegate BaseTrainer per fold (default DefaultTrainer),
    then aggregates the per-fold results into a KFoldResult.

    Depends only on BaseModelProxy/BaseDataPayload-shaped factories and
    sklearn.model_selection splitters, so it has zero torch import and works
    regardless of whether the torch extra is installed.

    Args:
        model_proxy_factory (Callable[[], BaseModelProxy]): Called once per fold;
            must return a freshly initialized model proxy (fresh weights/optimizer
            state) so folds don't leak information between each other.
        data_payload_factory (Callable[[np.ndarray, np.ndarray], BaseDataPayload]):
            Called once per fold with the exact (train_idx, val_idx) arrays that
            cv.split(X, y, groups) yields.
        X (Any): The data to split, passed through to cv.split.
        y (Optional[Any]): The target data, passed through to cv.split.
        groups (Optional[Any]): Group labels, passed through to cv.split.
        cv (Optional[BaseCrossValidator]): The cross-validation splitter. Defaults
            to KFold(n_splits=5, shuffle=True, random_state=random_state).
        random_state (Optional[int]): Random state used by the default splitter.
        trainer_factory (TrainerFactory): Builds the per-fold trainer. Defaults to
            a plain DefaultTrainer with no early stopping or callbacks.
        save_dir_path (Path): Root directory under which one collision-safe run
            directory is created; each fold gets a "fold_{i}" subdirectory within it.
        run_name (Optional[str]): Name for the shared root run directory. Defaults
            to a timestamp.
        logger (Optional[logging.Logger]): Logger for logging progress. Defaults to
            the module logger.
    """

    def __init__(
        self,
        model_proxy_factory: Callable[[], BaseModelProxy],
        data_payload_factory: Callable[[np.ndarray, np.ndarray], BaseDataPayload],
        X: Any,
        y: Any | None = None,
        groups: Any | None = None,
        cv: BaseCrossValidator | None = None,
        random_state: int | None = None,
        trainer_factory: TrainerFactory = _default_trainer_factory,
        save_dir_path: Path = DEFAULT_SAVE_PATH,
        run_name: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._model_proxy_factory = model_proxy_factory
        self._data_payload_factory = data_payload_factory
        self._X = X
        self._y = y
        self._groups = groups
        self._cv: BaseCrossValidator = cv or KFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
        self._trainer_factory = trainer_factory
        self._save_dir_path = save_dir_path
        self._run_name = run_name
        self._logger = logger or logging.getLogger(__name__)

        if not hasattr(self._cv, "split"):
            raise TrainerError("cv must be a splitter exposing a .split method.")

    def train(self, max_epochs: int) -> KFoldResult:
        """Train one model per cross-validation fold.

        Args:
            max_epochs (int): The maximum number of training epochs per fold.

        Returns:
            KFoldResult: The aggregate result across all folds.
        """
        if max_epochs <= 0:
            raise ValueError("Max epochs must be positive number greater than 0.")

        root_writer = CheckpointWriter(self._save_dir_path, run_name=self._run_name)
        root_writer.regenerate_model_save_path()
        root_dir = root_writer.get_model_save_path()

        fold_results = []
        model_name = ""
        for fold_index, (train_idx, val_idx) in enumerate(
            self._cv.split(self._X, self._y, self._groups), start=1
        ):
            model_proxy = self._model_proxy_factory()
            data_payload = self._data_payload_factory(train_idx, val_idx)

            fold_trainer = self._trainer_factory(
                model_proxy, data_payload, root_dir, f"fold_{fold_index}"
            )

            self._logger.info(f"Starting fold {fold_index}")
            fold_result = fold_trainer.train(max_epochs=max_epochs)
            fold_results.append(fold_result)
            model_name = fold_result.model_name

        return KFoldResult.from_fold_results(
            model_name=model_name, model_save_path=root_dir, fold_results=fold_results
        )
