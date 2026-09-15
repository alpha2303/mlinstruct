"""Binary classification example exercising ConfusionMatrix and ROC:
TorchDataPayload -> TorchModelProxy -> DefaultTrainer -> ConfusionMatrix / ROC.

Run with: uv run python examples/classification_metrics.py
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

from mlinstruct.evaluate.metrics.classification import ROC, ConfusionMatrix
from mlinstruct.train.data_payload import TorchDataPayload
from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.train.trainer import DefaultTrainer


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
    trainer = DefaultTrainer(
        model_proxy=proxy, data_payload=payload, save_dir_path=Path("examples_output")
    )
    result = trainer.train(max_epochs=15)
    print(f"Best validation loss: {result.best_val_loss:.4f}")

    y_true, y_logits = proxy.predict(val_loader)
    y_true_int = y_true.astype(int).ravel()
    y_scores = 1.0 / (1.0 + np.exp(-y_logits.ravel()))  # sigmoid: logits -> probabilities
    y_pred_int = (y_scores > 0.5).astype(int)

    cm = ConfusionMatrix.from_predictions(
        y_true_int, y_pred_int, class_labels=["class 0", "class 1"]
    )
    print("Confusion matrix:")
    print(cm.as_ndarray())

    roc = ROC.from_predictions(y_true_int, y_scores)

    fig, (cm_ax, roc_ax) = plt.subplots(1, 2, figsize=(10, 4))
    cm.plot(ax=cm_ax)
    roc.plot(ax=roc_ax)
    fig.tight_layout()

    plot_path = result.model_save_path / "classification_metrics.png"
    fig.savefig(plot_path)
    print(f"Confusion matrix + ROC plot saved to {plot_path}")


if __name__ == "__main__":
    main()
