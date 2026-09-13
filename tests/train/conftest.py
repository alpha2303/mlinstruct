import pytest

torch = pytest.importorskip("torch")

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split


@pytest.fixture(autouse=True)
def _seed_torch():
    torch.manual_seed(0)


@pytest.fixture
def tiny_model():
    return nn.Linear(4, 1)


@pytest.fixture
def tiny_loaders():
    inputs = torch.randn(64, 4)
    targets = inputs.sum(dim=1, keepdim=True) + 0.01 * torch.randn(64, 1)
    dataset = TensorDataset(inputs, targets)
    train_ds, val_ds, test_ds = random_split(dataset, [32, 16, 16])
    return (
        DataLoader(train_ds, batch_size=16),
        DataLoader(val_ds, batch_size=16),
        DataLoader(test_ds, batch_size=16),
    )


@pytest.fixture
def proxy(tiny_model):
    from mlinstruct.train.model_proxy import TorchModelProxy

    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    return TorchModelProxy(model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss())


@pytest.fixture
def payload(tiny_loaders):
    from mlinstruct.train.data_payload import TorchDataPayload

    train_loader, val_loader, test_loader = tiny_loaders
    return TorchDataPayload(
        train_data=train_loader, val_data=val_loader, test_data=test_loader
    )
