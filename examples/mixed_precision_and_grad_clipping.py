"""Mixed precision (autocast) and gradient clipping, both opt-in constructor
arguments on TorchModelProxy.

Run with: uv run python examples/mixed_precision_and_grad_clipping.py
"""

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from mlinstruct.train.data_payload import TorchDataPayload
from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.train.trainer import DefaultTrainer


def main() -> None:
    torch.manual_seed(0)

    inputs = torch.randn(512, 20)
    targets = inputs.sum(dim=1, keepdim=True) + 0.1 * torch.randn(512, 1)
    train_ds, val_ds = random_split(TensorDataset(inputs, targets), [400, 112])

    train_loader = DataLoader(train_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    proxy = TorchModelProxy(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        use_amp=True,  # bfloat16 on CPU, or CUDA when available
        max_grad_norm=1.0,  # clipped after unscaling, before each optimizer step
    )
    print(f"Training on device={proxy.device} with AMP autocast + grad-norm clipping enabled")

    payload = TorchDataPayload(train_data=train_loader, val_data=val_loader)
    trainer = DefaultTrainer(
        model_proxy=proxy, data_payload=payload, save_dir_path=Path("examples_output")
    )

    result = trainer.train(max_epochs=10)
    print(f"Trained for {result.epochs} epochs")
    print(f"Best validation loss: {result.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
