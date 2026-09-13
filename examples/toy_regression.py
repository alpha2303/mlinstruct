"""Runnable toy regression example exercising the full training pipeline:
TorchDataPayload -> TorchModelProxy -> DefaultTrainer -> LossPlotter.

Run with: uv run python examples/toy_regression.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from mlinstruct.evaluate.plots import LossPlotter
from mlinstruct.train.data_payload import TorchDataPayload
from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.train.trainer import DefaultTrainer


def main() -> None:
    torch.manual_seed(0)

    inputs = torch.randn(256, 4)
    targets = inputs.sum(dim=1, keepdim=True) + 0.1 * torch.randn(256, 1)
    dataset = TensorDataset(inputs, targets)
    train_ds, val_ds, test_ds = random_split(dataset, [180, 38, 38])

    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    model = nn.Linear(4, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    proxy = TorchModelProxy(model=model, optimizer=optimizer, loss_fn=nn.MSELoss())

    payload = TorchDataPayload(train_data=train_loader, val_data=val_loader, test_data=test_loader)
    trainer = DefaultTrainer(
        model_proxy=proxy, data_payload=payload, save_dir_path=Path("examples_output")
    )

    result = trainer.train(max_epochs=10)

    print(f"Trained for {result.epochs} epochs")
    print(f"Best validation loss: {result.best_val_loss:.4f}")
    print(f"Best checkpoint: {result.best_checkpoint_path}")

    _, ax = plt.subplots()
    LossPlotter().plot(ax, result.train_loss_list, result.val_loss_list)
    plot_path = result.model_save_path / "loss.png"
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")


if __name__ == "__main__":
    main()
