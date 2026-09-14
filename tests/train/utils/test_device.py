import pytest

torch = pytest.importorskip("torch")

from mlinstruct.train.utils.device import move_to_device, resolve_device  # noqa: E402


def test_resolve_device_explicit_string():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_explicit_device():
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_default_is_cpu_when_no_accelerator(mocker):
    mocker.patch.object(torch.accelerator, "current_accelerator", return_value=None)

    assert resolve_device() == torch.device("cpu")


def test_move_to_device_nested_structures():
    tensor = torch.tensor([1.0, 2.0])
    nested = {"a": (tensor, [tensor, tensor]), "b": "not a tensor"}

    moved = move_to_device(nested, torch.device("cpu"))

    assert moved["a"][0].device == torch.device("cpu")
    assert moved["a"][1][0].device == torch.device("cpu")
    assert moved["b"] == "not a tensor"
