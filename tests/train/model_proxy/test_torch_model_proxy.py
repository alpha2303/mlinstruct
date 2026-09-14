import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchinfo  # noqa: E402
from torch import nn, optim  # noqa: E402

from mlinstruct.train.model_proxy import TorchModelProxy  # noqa: E402
from mlinstruct.utils.exception import ModelProxyError  # noqa: E402

onnxruntime = pytest.importorskip("onnxruntime")


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


def test_validate_rejects_non_dataloader(proxy):
    with pytest.raises(ModelProxyError):
        proxy.validate([1, 2, 3])


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


def test_load_checkpoint_restores_scheduler_state(tiny_model, save_dir):
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    proxy = TorchModelProxy(
        model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss(), scheduler=scheduler
    )

    scheduler.step()
    saved_state = scheduler.state_dict()
    proxy.save_weights(epoch=1, save_dir_path=save_dir, model_name="ckpt", loss=0.1)

    scheduler.step()
    scheduler.step()
    assert scheduler.state_dict() != saved_state

    proxy.load_checkpoint(save_dir / "ckpt.pt")

    assert scheduler.state_dict() == saved_state


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


def test_scheduler_step_without_scheduler_raises(proxy):
    with pytest.raises(ModelProxyError):
        proxy.scheduler_step(avg_vloss=0.5)


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


def test_amp_disabled_by_default_has_no_scaler_effect(proxy):
    assert proxy._use_amp is False
    assert proxy._scaler.is_enabled() is False


def test_amp_cpu_bf16_trains(tiny_model, tiny_loaders):
    train_loader, _, _ = tiny_loaders
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(
        model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss(), use_amp=True
    )

    avg_loss = proxy.train_one_epoch(train_loader)

    assert proxy._amp_dtype == torch.bfloat16
    assert isinstance(avg_loss, float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_amp_cuda_fp16_uses_scaler(tiny_model, tiny_loaders, mocker):
    train_loader, _, _ = tiny_loaders
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(
        model=tiny_model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        device="cuda",
        use_amp=True,
        amp_dtype=torch.float16,
    )
    step_spy = mocker.spy(proxy._scaler, "step")

    proxy.train_one_epoch(train_loader)

    assert proxy._scaler.is_enabled() is True
    step_spy.assert_called()


def test_checkpoint_roundtrips_scaler_state(tiny_model, tiny_loaders, save_dir):
    train_loader, _, _ = tiny_loaders
    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    proxy = TorchModelProxy(
        model=tiny_model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        use_amp=True,
        amp_dtype=torch.float16,
    )
    proxy.train_one_epoch(train_loader)
    original_scaler_state = proxy._scaler.state_dict()

    proxy.save_weights(epoch=1, save_dir_path=save_dir, model_name="ckpt", loss=0.1)
    proxy._scaler = torch.amp.GradScaler(proxy._device.type, enabled=True)

    proxy.load_checkpoint(save_dir / "ckpt.pt")

    assert proxy._scaler.state_dict() == original_scaler_state


def test_grad_norm_is_clipped(tiny_model, tiny_loaders):
    train_loader, _, _ = tiny_loaders
    optimizer = optim.SGD(tiny_model.parameters(), lr=100.0)
    proxy = TorchModelProxy(
        model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss(), max_grad_norm=0.5
    )

    proxy.train_one_epoch(train_loader)

    grads = [p.grad.norm() for p in proxy._model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack(grads))

    assert total_norm.item() <= 0.5 + 1e-4


def test_no_clipping_when_none(proxy, tiny_loaders, mocker):
    train_loader, _, _ = tiny_loaders
    clip_spy = mocker.spy(nn.utils, "clip_grad_norm_")

    proxy.train_one_epoch(train_loader)

    clip_spy.assert_not_called()


def test_predict_shapes_and_no_grad(proxy, tiny_loaders):
    _, val_loader, _ = tiny_loaders

    y_true, y_pred = proxy.predict(val_loader)

    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert y_true.shape == y_pred.shape
    assert y_true.shape[0] == 16


def test_predict_rejects_non_dataloader(proxy):
    with pytest.raises(ModelProxyError):
        proxy.predict([1, 2, 3])


def test_lazy_import_returns_same_class_twice():
    from mlinstruct.train.model_proxy import TorchModelProxy as first_import
    from mlinstruct.train.model_proxy import TorchModelProxy as second_import

    assert first_import is second_import


def test_export_onnx_writes_file(proxy, save_dir):
    onnx_path = save_dir / "model.onnx"

    result_path = proxy.export_onnx(onnx_path, torch.randn(1, 4))

    assert result_path == onnx_path
    assert onnx_path.exists()


def test_export_onnx_output_matches_source_model(proxy, save_dir):
    onnx_path = save_dir / "model.onnx"
    input_sample = torch.randn(8, 4)

    proxy.export_onnx(onnx_path, input_sample)

    proxy._model.eval()
    with torch.no_grad():
        expected = proxy._model(input_sample).numpy()

    session = onnxruntime.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    (actual,) = session.run(None, {input_name: input_sample.numpy()})

    assert np.allclose(actual, expected, atol=1e-5)


def test_export_onnx_restores_training_mode(proxy, save_dir):
    input_sample = torch.randn(1, 4)

    proxy._model.train()
    proxy.export_onnx(save_dir / "train_mode.onnx", input_sample)
    assert proxy._model.training is True

    proxy._model.eval()
    proxy.export_onnx(save_dir / "eval_mode.onnx", input_sample)
    assert proxy._model.training is False


def test_export_onnx_rejects_missing_parent_dir(proxy, save_dir):
    missing_dir_path = save_dir / "does_not_exist" / "model.onnx"

    with pytest.raises(ModelProxyError):
        proxy.export_onnx(missing_dir_path, torch.randn(1, 4))


def test_export_onnx_without_onnx_extra_raises_import_error(proxy, save_dir, monkeypatch):
    import mlinstruct.utils.optional_deps as optional_deps

    real_find_spec = optional_deps.find_spec

    def fake_find_spec(name):
        return None if name == "onnx" else real_find_spec(name)

    monkeypatch.setattr(optional_deps, "find_spec", fake_find_spec)

    with pytest.raises(ImportError, match=r"mlinstruct\[onnx\]"):
        proxy.export_onnx(save_dir / "model.onnx", torch.randn(1, 4))
