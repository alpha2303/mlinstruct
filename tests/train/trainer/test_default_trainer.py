import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import optim  # noqa: E402

from mlinstruct.evaluate.callbacks import MetricsCallback  # noqa: E402
from mlinstruct.train.callbacks import TrainerCallback  # noqa: E402
from mlinstruct.train.data_payload import TorchDataPayload  # noqa: E402
from mlinstruct.train.model_proxy import TorchModelProxy  # noqa: E402
from mlinstruct.train.trainer import DefaultTrainer, default_trainer  # noqa: E402
from mlinstruct.utils.exception import TrainerError  # noqa: E402


class RecordingCallback(TrainerCallback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self, trainer) -> None:
        self.events.append("start")

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss) -> None:
        self.events.append(f"epoch_end:{epoch}")

    def on_train_end(self, trainer, result) -> None:
        self.events.append("end")


def test_scheduler_step_receives_val_loss(tiny_model, tiny_loaders, save_dir, mocker):
    train_loader, val_loader, _ = tiny_loaders
    data_payload = TorchDataPayload(train_data=train_loader, val_data=val_loader)

    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
    proxy = TorchModelProxy(
        model=tiny_model, optimizer=optimizer, loss_fn=torch.nn.MSELoss(), scheduler=scheduler
    )

    mocker.spy(proxy, "validate")
    mocker.spy(proxy, "scheduler_step")

    trainer = DefaultTrainer(model_proxy=proxy, data_payload=data_payload, save_dir_path=save_dir)
    trainer.train(max_epochs=1)

    proxy.scheduler_step.assert_called_once_with(avg_vloss=proxy.validate.spy_return)


def test_default_logger_is_module_logger(proxy, payload, save_dir):
    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)

    assert trainer._logger is logging.getLogger(default_trainer.__name__)


def test_train_returns_result_with_correct_epochs(proxy, payload, save_dir):
    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)

    result = trainer.train(max_epochs=2)

    assert result.epochs == 2


def test_train_loss_lists_have_len_epochs(proxy, payload, save_dir):
    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)

    result = trainer.train(max_epochs=2)

    assert len(result.train_loss_list) == 2
    assert len(result.val_loss_list) == 2


def test_train_writes_best_checkpoint(proxy, payload, save_dir):
    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)

    result = trainer.train(max_epochs=2)

    checkpoints = list(result.model_save_path.glob("*.pt"))
    assert len(checkpoints) >= 1

    checkpoint = torch.load(checkpoints[0], weights_only=False)
    assert "model_state_dict" in checkpoint


def test_train_rejects_non_positive_epochs(proxy, payload, save_dir):
    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)

    with pytest.raises(ValueError):
        trainer.train(max_epochs=0)


def test_early_stop_breaks_loop(proxy, payload, save_dir, mocker):
    mocker.patch.object(proxy, "train_one_epoch", return_value=0.5)
    mocker.patch.object(proxy, "validate", return_value=1.0)

    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)
    trainer.add_early_stop(patience=2, min_delta=0.01)

    result = trainer.train(max_epochs=10)

    assert result.epochs == 3
    assert len(result.train_loss_list) == 3


def test_test_set_evaluated_when_present(proxy, payload, save_dir, mocker):
    validate_spy = mocker.spy(proxy, "validate")

    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)
    trainer.train(max_epochs=1)

    test_loader = payload.get_test_data()
    assert any(call.args and call.args[0] is test_loader for call in validate_spy.call_args_list)


def test_constructor_rejects_wrong_types(proxy, payload, save_dir):
    with pytest.raises(TrainerError):
        DefaultTrainer(model_proxy=object(), data_payload=payload, save_dir_path=save_dir)  # type: ignore

    with pytest.raises(TrainerError):
        DefaultTrainer(model_proxy=proxy, data_payload=object(), save_dir_path=save_dir)  # type: ignore


def test_train_result_reports_best_checkpoint_path_that_exists(proxy, payload, save_dir):
    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)

    result = trainer.train(max_epochs=2)

    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.exists()
    assert result.best_val_loss == min(result.val_loss_list)


def test_stopped_early_flag_set_when_early_stopper_triggers(proxy, payload, save_dir, mocker):
    mocker.patch.object(proxy, "train_one_epoch", return_value=0.5)
    mocker.patch.object(proxy, "validate", return_value=1.0)

    trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, save_dir_path=save_dir)
    trainer.add_early_stop(patience=2, min_delta=0.01)

    result = trainer.train(max_epochs=10)

    assert result.stopped_early is True


def test_resume_continues_from_saved_epoch(tiny_model, tiny_loaders, save_dir, mocker):
    train_loader, val_loader, _ = tiny_loaders
    data_payload = TorchDataPayload(train_data=train_loader, val_data=val_loader)

    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(model=tiny_model, optimizer=optimizer, loss_fn=torch.nn.MSELoss())

    mocker.patch.object(proxy, "train_one_epoch", side_effect=[0.5, 0.4, 0.3, 0.2])
    mocker.patch.object(proxy, "validate", side_effect=[0.9, 0.8, 0.7, 0.6])

    trainer = DefaultTrainer(model_proxy=proxy, data_payload=data_payload, save_dir_path=save_dir)
    first_result = trainer.train(max_epochs=2)

    assert first_result.epochs == 2
    assert len(first_result.train_loss_list) == 2

    checkpoint_path = sorted(first_result.model_save_path.glob("*.pt"))[-1]

    second_result = trainer.train(max_epochs=4, resume_from=checkpoint_path)

    assert second_result.epochs == 4
    assert len(second_result.train_loss_list) == 2


def test_show_progress_without_tqdm_does_not_raise(proxy, payload, save_dir, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("simulated missing tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    trainer = DefaultTrainer(
        model_proxy=proxy, data_payload=payload, save_dir_path=save_dir, show_progress=True
    )

    result = trainer.train(max_epochs=1)

    assert result.epochs == 1


def test_callbacks_invoked_in_order(proxy, payload, save_dir):
    callback = RecordingCallback()
    trainer = DefaultTrainer(
        model_proxy=proxy, data_payload=payload, save_dir_path=save_dir, callbacks=[callback]
    )

    trainer.train(max_epochs=2)

    assert callback.events == ["start", "epoch_end:1", "epoch_end:2", "end"]


def test_metrics_callback_history_length_equals_epochs(tiny_model, tiny_loaders, save_dir):
    train_loader, val_loader, _ = tiny_loaders
    data_payload = TorchDataPayload(train_data=train_loader, val_data=val_loader)
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(model=tiny_model, optimizer=optimizer, loss_fn=torch.nn.MSELoss())

    callback = MetricsCallback(
        metrics={"mae": lambda y, y_pred: float(np.mean(np.abs(y - y_pred)))},
        loader=val_loader,
    )
    trainer = DefaultTrainer(
        model_proxy=proxy, data_payload=data_payload, save_dir_path=save_dir, callbacks=[callback]
    )

    trainer.train(max_epochs=3)

    assert len(callback.history["mae"]) == 3


def test_metrics_history_in_train_result(tiny_model, tiny_loaders, save_dir):
    train_loader, val_loader, _ = tiny_loaders
    data_payload = TorchDataPayload(train_data=train_loader, val_data=val_loader)
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(model=tiny_model, optimizer=optimizer, loss_fn=torch.nn.MSELoss())

    callback = MetricsCallback(
        metrics={"mae": lambda y, y_pred: float(np.mean(np.abs(y - y_pred)))},
        loader=val_loader,
    )
    trainer = DefaultTrainer(
        model_proxy=proxy, data_payload=data_payload, save_dir_path=save_dir, callbacks=[callback]
    )

    result = trainer.train(max_epochs=2)

    assert result.metrics_history == callback.history
