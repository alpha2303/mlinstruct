import pytest

from mlinstruct.train.data_payload import TorchDataPayload


def test_valid_construction(tiny_loaders):
    train_loader, val_loader, test_loader = tiny_loaders

    payload = TorchDataPayload(
        train_data=train_loader, val_data=val_loader, test_data=test_loader
    )

    assert payload.get_train_data() is train_loader
    assert payload.get_val_data() is val_loader
    assert payload.get_test_data() is test_loader


def test_has_test_data_true_when_provided(tiny_loaders):
    train_loader, val_loader, test_loader = tiny_loaders

    payload = TorchDataPayload(
        train_data=train_loader, val_data=val_loader, test_data=test_loader
    )

    assert payload.has_test_data() is True


def test_has_test_data_false_when_absent(tiny_loaders):
    train_loader, val_loader, _ = tiny_loaders

    payload = TorchDataPayload(train_data=train_loader, val_data=val_loader)

    assert payload.has_test_data() is False


def test_rejects_non_dataloader_train_data(tiny_loaders):
    _, val_loader, test_loader = tiny_loaders

    with pytest.raises(TypeError):
        TorchDataPayload(train_data=[1, 2, 3], val_data=val_loader, test_data=test_loader)


def test_rejects_non_dataloader_val_data(tiny_loaders):
    train_loader, _, test_loader = tiny_loaders

    with pytest.raises(TypeError):
        TorchDataPayload(train_data=train_loader, val_data=[1, 2, 3], test_data=test_loader)


def test_rejects_non_dataloader_test_data(tiny_loaders):
    train_loader, val_loader, _ = tiny_loaders

    with pytest.raises(TypeError):
        TorchDataPayload(train_data=train_loader, val_data=val_loader, test_data=[1, 2, 3])
