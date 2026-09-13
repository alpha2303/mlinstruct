import pytest
import torch
import torchinfo
from torch import nn, optim

from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.utils.exception import ModelProxyError


def test_default_model_name(proxy, tiny_model):
    assert proxy.get_model_name() == type(tiny_model).__name__


def test_custom_model_name(tiny_model):
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    named_proxy = TorchModelProxy(
        model=tiny_model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        model_name="my-model",
    )

    assert named_proxy.get_model_name() == "my-model"


def test_train_one_epoch_reduces_loss(proxy, tiny_loaders):
    train_loader, _, _ = tiny_loaders

    first_epoch_loss = proxy.train_one_epoch(train_loader)
    second_epoch_loss = proxy.train_one_epoch(train_loader)

    assert second_epoch_loss < first_epoch_loss


def test_train_one_epoch_rejects_non_dataloader(proxy):
    with pytest.raises(ModelProxyError):
        proxy.train_one_epoch([1, 2, 3])


def test_validate_does_not_update_weights(proxy, tiny_loaders):
    _, val_loader, _ = tiny_loaders
    before = {key: value.clone() for key, value in proxy._model.state_dict().items()}

    proxy.validate(val_loader)

    after = proxy._model.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key])


def test_save_then_load_checkpoint_roundtrip(proxy, save_dir):
    original_state = {key: value.clone() for key, value in proxy._model.state_dict().items()}
    original_lr = proxy._optimizer.state_dict()["param_groups"][0]["lr"]

    proxy.save_weights(epoch=3, save_dir_path=save_dir, model_name="ckpt", loss=0.1)
    checkpoint_path = save_dir / "ckpt.pt"

    with torch.no_grad():
        for param in proxy._model.parameters():
            param.zero_()

    loaded_epoch = proxy.load_checkpoint(checkpoint_path)

    assert loaded_epoch == 3
    for key, value in original_state.items():
        assert torch.equal(proxy._model.state_dict()[key], value)
    assert proxy._optimizer.state_dict()["param_groups"][0]["lr"] == original_lr


def test_checkpoint_contains_scheduler_state_when_present(tiny_model, save_dir):
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
    proxy = TorchModelProxy(
        model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss(), scheduler=scheduler
    )

    proxy.save_weights(epoch=1, save_dir_path=save_dir, model_name="ckpt", loss=0.1)
    checkpoint = torch.load(save_dir / "ckpt.pt", weights_only=True)

    assert checkpoint["scheduler_state_dict"] is not None


def test_checkpoint_contains_none_scheduler_state_when_absent(proxy, save_dir):
    proxy.save_weights(epoch=1, save_dir_path=save_dir, model_name="ckpt", loss=0.1)
    checkpoint = torch.load(save_dir / "ckpt.pt", weights_only=True)

    assert checkpoint["scheduler_state_dict"] is None


def test_load_checkpoint_returns_epoch_and_keeps_train_mode(proxy, save_dir):
    proxy.save_weights(epoch=5, save_dir_path=save_dir, model_name="ckpt", loss=0.1)
    proxy._model.eval()

    loaded_epoch = proxy.load_checkpoint(save_dir / "ckpt.pt")

    assert loaded_epoch == 5
    assert proxy._model.training is False


def test_load_checkpoint_uses_weights_only(proxy, save_dir, mocker):
    proxy.save_weights(epoch=1, save_dir_path=save_dir, model_name="ckpt", loss=0.1)
    load_spy = mocker.spy(torch, "load")

    proxy.load_checkpoint(save_dir / "ckpt.pt")

    assert load_spy.call_args.kwargs["weights_only"] is True


def test_load_checkpoint_map_location_matches_device(proxy, save_dir, mocker):
    proxy.save_weights(epoch=1, save_dir_path=save_dir, model_name="ckpt", loss=0.1)
    load_spy = mocker.spy(torch, "load")

    proxy.load_checkpoint(save_dir / "ckpt.pt")

    assert load_spy.call_args.kwargs["map_location"] == proxy.device


def test_load_weights_is_deprecated_alias_for_load_checkpoint(proxy, save_dir):
    proxy.save_weights(epoch=7, save_dir_path=save_dir, model_name="ckpt", loss=0.1)

    with pytest.deprecated_call():
        proxy.load_weights(save_dir / "ckpt.pt")


def test_save_weights_rejects_missing_dir(proxy, save_dir):
    missing_dir = save_dir / "does_not_exist"

    with pytest.raises(ModelProxyError):
        proxy.save_weights(epoch=1, save_dir_path=missing_dir, model_name="ckpt", loss=0.1)


def test_has_scheduler_false_by_default(proxy):
    assert proxy.has_scheduler() is False


def test_scheduler_step_plateau_vs_step(tiny_model, mocker):
    plateau_optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(plateau_optimizer)
    plateau_proxy = TorchModelProxy(
        model=tiny_model,
        optimizer=plateau_optimizer,
        loss_fn=nn.MSELoss(),
        scheduler=plateau_scheduler,
    )
    plateau_step_spy = mocker.spy(plateau_scheduler, "step")

    plateau_proxy.scheduler_step(avg_vloss=0.5)

    plateau_step_spy.assert_called_once_with(0.5)

    step_optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    step_scheduler = optim.lr_scheduler.StepLR(step_optimizer, step_size=1)
    step_proxy = TorchModelProxy(
        model=tiny_model,
        optimizer=step_optimizer,
        loss_fn=nn.MSELoss(),
        scheduler=step_scheduler,
    )
    step_spy = mocker.spy(step_scheduler, "step")

    step_proxy.scheduler_step(avg_vloss=0.5)

    step_spy.assert_called_once_with()


def test_summary_returns_model_statistics(proxy):
    assert isinstance(proxy.summary(), torchinfo.ModelStatistics)


def test_model_is_on_requested_device(tiny_model):
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(
        model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss(), device="cpu"
    )

    assert proxy.device == torch.device("cpu")
    assert next(proxy._model.parameters()).device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_train_one_epoch_on_cuda(tiny_model, tiny_loaders):
    train_loader, _, _ = tiny_loaders
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(
        model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss(), device="cuda"
    )

    proxy.train_one_epoch(train_loader)

    assert next(proxy._model.parameters()).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_train_with_cpu_loader_and_gpu_model_does_not_raise(tiny_model, tiny_loaders):
    train_loader, _, _ = tiny_loaders
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(
        model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss(), device="cuda"
    )

    proxy.train_one_epoch(train_loader)


def test_lazy_import_returns_same_class_twice():
    from mlinstruct.train.model_proxy import TorchModelProxy as first_import
    from mlinstruct.train.model_proxy import TorchModelProxy as second_import

    assert first_import is second_import
