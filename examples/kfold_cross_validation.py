"""K-Fold cross-validation via KFoldTrainer: trains one independent model per
fold and aggregates results into a KFoldResult.

Run with: uv run python examples/kfold_cross_validation.py
"""

from pathlib import Path

import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import TensorDataset

from mlinstruct.train.data_payload import torch_kfold_data_payload
from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.train.trainer import KFoldTrainer


def model_proxy_factory() -> TorchModelProxy:
    """Called once per fold; must return a freshly initialized proxy so folds
    don't leak weights/optimizer state between each other."""
    model = nn.Linear(4, 1)
    return TorchModelProxy(
        model=model, optimizer=optim.SGD(model.parameters(), lr=0.01), loss_fn=nn.MSELoss()
    )


def main() -> None:
    torch.manual_seed(0)

    inputs = torch.randn(256, 4)
    targets = inputs.sum(dim=1, keepdim=True) + 0.1 * torch.randn(256, 1)
    dataset = TensorDataset(inputs, targets)

    def data_payload_factory(train_idx, val_idx):
        # functools.partial(torch_kfold_data_payload, dataset=dataset, batch_size=16)
        # doesn't work here: KFoldTrainer calls factory(train_idx, val_idx)
        # positionally, and those positional args would collide with the
        # keyword-bound `dataset` (torch_kfold_data_payload's first parameter).
        # A small wrapper that passes everything by keyword sidesteps that.
        return torch_kfold_data_payload(
            dataset=dataset, train_idx=train_idx, val_idx=val_idx, batch_size=16
        )

    kfold_trainer = KFoldTrainer(
        model_proxy_factory=model_proxy_factory,
        data_payload_factory=data_payload_factory,
        X=inputs,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        save_dir_path=Path("examples_output"),
    )

    kfold_result = kfold_trainer.train(max_epochs=15)

    print(f"Mean val loss: {kfold_result.mean_val_loss:.4f} +/- {kfold_result.std_val_loss:.4f}")
    print(f"Best fold: {kfold_result.best_fold_index}")
    for i, fold_result in enumerate(kfold_result.fold_results, start=1):
        print(f"  fold {i}: best_val_loss={fold_result.best_val_loss:.4f}")
    print(f"Checkpoints under: {kfold_result.model_save_path}")


if __name__ == "__main__":
    main()
