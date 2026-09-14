import pytest

from mlinstruct.utils.optional_deps import is_installed


@pytest.fixture(autouse=True)
def _seed_torch():
    if is_installed("torch"):
        import torch

        torch.manual_seed(0)


@pytest.fixture
def tiny_model():
    pytest.importorskip("torch")
    from torch import nn

    return nn.Linear(4, 1)


@pytest.fixture
def tiny_loaders():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset, random_split

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
    pytest.importorskip("torch")
    from torch import nn, optim

    from mlinstruct.train.model_proxy import TorchModelProxy

    optimizer = optim.SGD(tiny_model.parameters(), lr=0.01)
    return TorchModelProxy(model=tiny_model, optimizer=optimizer, loss_fn=nn.MSELoss())


@pytest.fixture
def payload(tiny_loaders):
    pytest.importorskip("torch")
    from mlinstruct.train.data_payload import TorchDataPayload

    train_loader, val_loader, test_loader = tiny_loaders
    return TorchDataPayload(train_data=train_loader, val_data=val_loader, test_data=test_loader)


@pytest.fixture
def tiny_generator():
    pytest.importorskip("torch")
    from torch import nn

    return nn.Sequential(nn.Linear(3, 4), nn.Tanh())


@pytest.fixture
def tiny_discriminator():
    pytest.importorskip("torch")
    from torch import nn

    return nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())


@pytest.fixture
def gan_proxy(tiny_generator, tiny_discriminator):
    pytest.importorskip("torch")
    from torch import optim

    from mlinstruct.train.model_proxy import GANModelProxy

    return GANModelProxy(
        generator=tiny_generator,
        discriminator=tiny_discriminator,
        generator_optimizer=optim.SGD(tiny_generator.parameters(), lr=0.01),
        discriminator_optimizer=optim.SGD(tiny_discriminator.parameters(), lr=0.01),
        latent_dim=3,
    )


@pytest.fixture
def real_samples_loader():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.randn(32, 4))
    return DataLoader(dataset, batch_size=8)
