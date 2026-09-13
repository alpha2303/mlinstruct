import logging

import pytest
import torch
from torch import optim

from mlinstruct.train.data_payload import TorchDataPayload
from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.train.trainer import DefaultTrainer
from mlinstruct.train.trainer import default_trainer
from mlinstruct.utils.exception import TrainerError


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

    trainer = DefaultTrainer(
        model_proxy=proxy, data_payload=data_payload, save_dir_path=save_dir
    )
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
    assert any(
        call.args and call.args[0] is test_loader
        for call in validate_spy.call_args_list
    )


def test_constructor_rejects_wrong_types(proxy, payload, save_dir):
    with pytest.raises(TrainerError):
        DefaultTrainer(model_proxy=object(), data_payload=payload, save_dir_path=save_dir)  # type: ignore

    with pytest.raises(TrainerError):
        DefaultTrainer(model_proxy=proxy, data_payload=object(), save_dir_path=save_dir)  # type: ignore
