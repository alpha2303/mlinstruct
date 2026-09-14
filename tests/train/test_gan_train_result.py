from pathlib import Path

from mlinstruct.train import GANTrainResult


def test_field_round_trip():
    result = GANTrainResult(
        model_name="tiny-gan",
        model_save_path=Path("./Models/run"),
        epochs=3,
        g_loss_list=[0.9, 0.8, 0.7],
        d_loss_list=[1.2, 1.1, 1.0],
    )

    assert result.model_name == "tiny-gan"
    assert result.model_save_path == Path("./Models/run")
    assert result.epochs == 3
    assert result.g_loss_list == [0.9, 0.8, 0.7]
    assert result.d_loss_list == [1.2, 1.1, 1.0]
    assert result.metrics_history == {}
