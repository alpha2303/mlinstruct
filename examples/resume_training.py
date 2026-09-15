"""Resuming training from a checkpoint. Trains to epoch 10, then continues the
same run up to epoch 20 by passing the first phase's best checkpoint as
resume_from -- max_epochs stays the absolute epoch to train up to, not an
additional count.

Run with: uv run python examples/resume_training.py
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

    inputs = torch.randn(256, 4)
    targets = inputs.sum(dim=1, keepdim=True) + 0.1 * torch.randn(256, 1)
    train_ds, val_ds = random_split(TensorDataset(inputs, targets), [200, 56])

    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = nn.Linear(4, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    proxy = TorchModelProxy(model=model, optimizer=optimizer, loss_fn=nn.MSELoss())
    payload = TorchDataPayload(train_data=train_loader, val_data=val_loader)

    trainer = DefaultTrainer(
        model_proxy=proxy,
        data_payload=payload,
        save_dir_path=Path("examples_output"),
        run_name="resume_demo",
    )

    phase_one = trainer.train(max_epochs=10)
    print(
        f"Phase 1: trained to epoch {phase_one.epochs}, best val loss {phase_one.best_val_loss:.4f}"
    )
    print(f"Phase 1 checkpoint: {phase_one.best_checkpoint_path}")

    # Each train() call gets its own fresh run directory (resume_demo,
    # resume_demo_1, ...); resume_from points at the checkpoint file directly,
    # so which directory it lives in doesn't matter.
    phase_two = trainer.train(max_epochs=20, resume_from=phase_one.best_checkpoint_path)
    print(
        f"Phase 2: trained to epoch {phase_two.epochs}, best val loss {phase_two.best_val_loss:.4f}"
    )
    print(f"Phase 2 checkpoint: {phase_two.best_checkpoint_path}")


if __name__ == "__main__":
    main()
