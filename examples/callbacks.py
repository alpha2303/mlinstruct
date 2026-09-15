"""Trainer callbacks example: a custom TrainerCallback plus the built-in
MetricsCallback, tracking validation accuracy every epoch.

Run with: uv run python examples/callbacks.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from mlinstruct.evaluate.callbacks import MetricsCallback, accuracy
from mlinstruct.train.callbacks import TrainerCallback
from mlinstruct.train.data_payload import TorchDataPayload
from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.train.trainer import DefaultTrainer


class PrintOnEpochEnd(TrainerCallback):
    """A custom callback: logs a one-line summary after every epoch."""

    def on_train_start(self, trainer) -> None:
        print("Training started")

    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: float) -> None:
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    def on_train_end(self, trainer, result) -> None:
        print(f"Training finished after {result.epochs} epochs")


def logits_accuracy(y_true: np.ndarray, y_logits: np.ndarray) -> float:
    """Wrap accuracy() so MetricsCallback can score against raw model output:
    threshold the logit at 0 (equivalent to a 0.5 probability cutoff)."""
    y_true_int = y_true.astype(int).ravel()
    y_pred_int = (y_logits.ravel() > 0).astype(int)
    return accuracy(y_true_int, y_pred_int)


def main() -> None:
    torch.manual_seed(0)

    X, y = make_blobs(n_samples=400, centers=2, cluster_std=2.0, random_state=0)
    inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(inputs, targets)
    train_ds, val_ds = random_split(dataset, [320, 80])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    proxy = TorchModelProxy(model=model, optimizer=optimizer, loss_fn=nn.BCEWithLogitsLoss())
    payload = TorchDataPayload(train_data=train_loader, val_data=val_loader)

    metrics_callback = MetricsCallback(metrics={"accuracy": logits_accuracy}, loader=val_loader)
    trainer = DefaultTrainer(
        model_proxy=proxy,
        data_payload=payload,
        save_dir_path=Path("examples_output"),
        callbacks=[PrintOnEpochEnd(), metrics_callback],
    )

    result = trainer.train(max_epochs=15)

    print(f"Validation accuracy history: {result.metrics_history['accuracy']}")

    _, ax = plt.subplots()
    ax.plot(result.metrics_history["accuracy"])
    ax.set_title("Validation accuracy per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    plot_path = result.model_save_path / "accuracy.png"
    plt.savefig(plot_path)
    print(f"Accuracy plot saved to {plot_path}")


if __name__ == "__main__":
    main()
