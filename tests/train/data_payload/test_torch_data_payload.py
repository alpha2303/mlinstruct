import numpy as np
import pytest

pytest.importorskip("torch")

from mlinstruct.train.data_payload import TorchDataPayload, torch_kfold_data_payload  # noqa: E402


def test_valid_construction(tiny_loaders):
    train_loader, val_loader, test_loader = tiny_loaders

    payload = TorchDataPayload(train_data=train_loader, val_data=val_loader, test_data=test_loader)

    assert payload.get_train_data() is train_loader
    assert payload.get_val_data() is val_loader
    assert payload.get_test_data() is test_loader


def test_has_test_data_true_when_provided(tiny_loaders):
    train_loader, val_loader, test_loader = tiny_loaders

    payload = TorchDataPayload(train_data=train_loader, val_data=val_loader, test_data=test_loader)

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


def test_torch_kfold_data_payload_builds_subset_loaders_of_correct_length():
    import torch
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.randn(10, 4), torch.randn(10, 1))
    train_idx = np.arange(0, 8)
    val_idx = np.arange(8, 10)

    payload = torch_kfold_data_payload(
        dataset=dataset, train_idx=train_idx, val_idx=val_idx, batch_size=4
    )

    assert sum(len(batch[0]) for batch in payload.get_train_data()) == 8
    assert sum(len(batch[0]) for batch in payload.get_val_data()) == 2


def test_torch_kfold_data_payload_passes_through_dataloader_kwargs():
    import torch
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.randn(10, 4), torch.randn(10, 1))

    payload = torch_kfold_data_payload(
        dataset=dataset,
        train_idx=np.arange(0, 8),
        val_idx=np.arange(8, 10),
        batch_size=3,
        drop_last=True,
    )

    train_batches = list(payload.get_train_data())
    assert all(len(batch[0]) == 3 for batch in train_batches)


def test_torch_kfold_data_payload_includes_test_data_when_given():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.randn(10, 4), torch.randn(10, 1))
    test_loader = DataLoader(TensorDataset(torch.randn(4, 4), torch.randn(4, 1)), batch_size=2)

    payload = torch_kfold_data_payload(
        dataset=dataset,
        train_idx=np.arange(0, 8),
        val_idx=np.arange(8, 10),
        batch_size=4,
        test_data=test_loader,
    )

    assert payload.has_test_data() is True
    assert payload.get_test_data() is test_loader
