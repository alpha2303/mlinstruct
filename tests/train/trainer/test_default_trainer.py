import logging

import torch
from torch import optim

from mlinstruct.train.data_payload import TorchDataPayload
from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.train.trainer import DefaultTrainer
from mlinstruct.train.trainer import default_trainer


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
