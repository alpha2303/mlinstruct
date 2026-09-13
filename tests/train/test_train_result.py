from pathlib import Path

from mlinstruct.train import TrainResult


def test_field_round_trip():
    result = TrainResult(
        model_name="tiny",
        model_save_path=Path("./Models/run"),
        epochs=3,
        train_loss_list=[0.5, 0.3, 0.2],
        val_loss_list=[0.6, 0.4, 0.25],
    )

    assert result.model_name == "tiny"
    assert result.model_save_path == Path("./Models/run")
    assert result.epochs == 3
    assert result.train_loss_list == [0.5, 0.3, 0.2]
    assert result.val_loss_list == [0.6, 0.4, 0.25]
