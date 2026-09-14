from pathlib import Path

import numpy as np

from mlinstruct.train import KFoldResult, TrainResult


def _make_result(best_val_loss: float) -> TrainResult:
    return TrainResult(
        model_name="tiny",
        model_save_path=Path("./Models/run/fold_1"),
        epochs=2,
        train_loss_list=[0.5, 0.4],
        val_loss_list=[0.6, best_val_loss],
        best_val_loss=best_val_loss,
        best_checkpoint_path=None,
        stopped_early=False,
    )


def test_from_fold_results_computes_mean_and_std():
    fold_results = [_make_result(0.5), _make_result(0.3), _make_result(0.4)]

    result = KFoldResult.from_fold_results(
        model_name="tiny", model_save_path=Path("./Models/run"), fold_results=fold_results
    )

    assert result.mean_val_loss == np.mean([0.5, 0.3, 0.4])
    assert result.std_val_loss == np.std([0.5, 0.3, 0.4])


def test_best_fold_index_matches_lowest_best_val_loss():
    fold_results = [_make_result(0.5), _make_result(0.2), _make_result(0.4)]

    result = KFoldResult.from_fold_results(
        model_name="tiny", model_save_path=Path("./Models/run"), fold_results=fold_results
    )

    assert result.best_fold_index == 2


def test_field_round_trip():
    fold_results = [_make_result(0.5)]
    result = KFoldResult(
        model_name="tiny",
        model_save_path=Path("./Models/run"),
        fold_results=fold_results,
        mean_val_loss=0.5,
        std_val_loss=0.0,
        best_fold_index=1,
    )

    assert result.model_name == "tiny"
    assert result.model_save_path == Path("./Models/run")
    assert result.fold_results == fold_results
    assert result.mean_val_loss == 0.5
    assert result.std_val_loss == 0.0
    assert result.best_fold_index == 1
