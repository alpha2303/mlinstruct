# mlinstruct

A small, readable PyTorch training loop and evaluation toolkit — not a
Lightning competitor. `mlinstruct` gives you a `Trainer` you can actually
read end to end in a few minutes: a data payload, a model proxy, a training
loop with checkpointing and early stopping, and a handful of evaluation
metrics and plots. Drop it into a personal project instead of writing the
same training loop from scratch again.

The core (`evaluate`, `TrainResult`, `EarlyStopper`, `CheckpointWriter`) is
framework-agnostic and has no dependency on PyTorch. The `TorchModelProxy` /
`TorchDataPayload` implementations, currently the only backend, live behind
an optional extra.

## Install

```bash
# Core only (no torch)
pip install mlinstruct

# With the PyTorch backend, CPU build
pip install mlinstruct[torch]

# ... plus ONNX export support
pip install mlinstruct[torch,onnx]
```

For a CUDA build, use the `torch-cuda` extra instead of `torch` — it resolves
`torch` from PyTorch's cu126 index automatically (via `uv`'s per-extra
`tool.uv.sources`, already configured in this project's `pyproject.toml`):

```bash
uv add mlinstruct[torch-cuda]
```

`torch` and `torch-cuda` are declared as conflicting extras, so `uv` refuses
to install both at once. This only works through `uv`; plain `pip install
mlinstruct[torch-cuda]` installs the same PyPI `torch` as `mlinstruct[torch]`
since pip has no notion of `tool.uv.sources` — pip users who need the cu126
build should pass `--index-url https://download.pytorch.org/whl/cu126`
themselves.

Requires Python >= 3.12. `mlinstruct[torch]` / `mlinstruct[torch-cuda]`
require `torch>=2.6`.

## Quickstart

```python
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from mlinstruct.train.data_payload import TorchDataPayload
from mlinstruct.train.model_proxy import TorchModelProxy
from mlinstruct.train.trainer import DefaultTrainer
from mlinstruct.evaluate.plots import LossPlotter

inputs = torch.randn(256, 4)
targets = inputs.sum(dim=1, keepdim=True) + 0.1 * torch.randn(256, 1)
train_ds, val_ds = random_split(TensorDataset(inputs, targets), [200, 56])

payload = TorchDataPayload(
    train_data=DataLoader(train_ds, batch_size=16),
    val_data=DataLoader(val_ds, batch_size=16),
)

model = nn.Linear(4, 1)
proxy = TorchModelProxy(
    model=model,
    optimizer=optim.SGD(model.parameters(), lr=0.01),
    loss_fn=nn.MSELoss(),
)

trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload)
trainer.add_early_stop(patience=3)

result = trainer.train(max_epochs=20)
print(f"Best val loss: {result.best_val_loss:.4f}, checkpoint: {result.best_checkpoint_path}")

_, ax = plt.subplots()
LossPlotter().plot(ax, result.train_loss_list, result.val_loss_list)
```

For a classification model, evaluate with `ConfusionMatrix` instead:

```python
from mlinstruct.evaluate.metrics.classification import ConfusionMatrix

y_true, y_pred = proxy.predict(payload.get_val_data())
ConfusionMatrix.from_predictions(y_true, y_pred.argmax(axis=1)).plot()
```

See [`examples/toy_regression.py`](examples/toy_regression.py) for a
complete, runnable script.

## Resuming training

Every call to `train()` writes checkpoints under a per-run directory. Pass
the path to a checkpoint's `resume_from` to pick up where it left off —
`max_epochs` stays the absolute epoch to train up to, not an additional
count:

```python
result = trainer.train(max_epochs=10)
# ...later, in a new process...
result = trainer.train(max_epochs=20, resume_from=result.best_checkpoint_path)
```

## Mixed precision and gradient clipping

Both are opt-in constructor arguments on `TorchModelProxy`:

```python
proxy = TorchModelProxy(
    model=model,
    optimizer=optimizer,
    loss_fn=nn.MSELoss(),
    use_amp=True,  # bfloat16 on CPU, or CUDA when available
    max_grad_norm=1.0,  # clipped after unscaling, before each optimizer step
)
```

## Callbacks

Subclass `TrainerCallback` to hook into training, or use the built-in
`MetricsCallback` to track a metric on a held-out loader every epoch:

```python
from mlinstruct.evaluate.callbacks import MetricsCallback, accuracy

metrics = MetricsCallback(metrics={"accuracy": accuracy}, loader=val_loader)
trainer = DefaultTrainer(model_proxy=proxy, data_payload=payload, callbacks=[metrics])

result = trainer.train(max_epochs=10)
print(result.metrics_history["accuracy"])
```

```python
from mlinstruct.train.callbacks import TrainerCallback


class PrintOnEpochEnd(TrainerCallback):
    def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
        print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
```

## ONNX export

`TorchModelProxy` implements `OnnxExportable`, a backend-agnostic capability
interface (`isinstance(proxy, OnnxExportable)`) that any future `ModelProxy`
can opt into on its own terms. Requires the `onnx` extra:

```python
proxy.export_onnx(Path("model.onnx"), input_sample=torch.randn(1, 4))
```

`input_sample` is a representative single batch used for tracing/shape
inference. Extra keyword arguments (`input_names`, `dynamic_shapes`,
`opset_version`, ...) pass straight through to `torch.onnx.export`; mlinstruct
deliberately doesn't pin its own names for these since torch's exporter API
has been shifting release to release. Pass `dynamo=False` as an escape hatch
for models that don't trace cleanly under the default dynamo-based exporter.

## API overview

| Class | Module | Purpose |
|---|---|---|
| `TorchDataPayload` | `train.data_payload` | Wraps train/val/test `DataLoader`s. |
| `TorchModelProxy` | `train.model_proxy` | Adapts a model, optimizer, loss, and optional scheduler to the trainer; owns device placement, AMP, and gradient clipping. |
| `OnnxExportable` | `train.model_proxy` | Opt-in capability interface for backends that can export to ONNX; `TorchModelProxy` implements it. |
| `DefaultTrainer` | `train.trainer` | Runs the epoch loop: training, validation, scheduler step, checkpointing, early stopping, callbacks. |
| `EarlyStopper` | `train.utils` | Stops training when validation loss plateaus. |
| `CheckpointWriter` | `train.utils` | Writes checkpoints to a unique, per-run directory. |
| `TrainResult` | `train` | Frozen dataclass returned by `trainer.train(...)`. |
| `TrainerCallback` | `train.callbacks` | Base class for `on_train_start` / `on_epoch_end` / `on_train_end` hooks. |
| `MetricsCallback` | `evaluate.callbacks` | Computes named metrics on a loader every epoch. |
| `ConfusionMatrix`, `ROC` | `evaluate.metrics.classification` | Classification metrics with a `.plot()`. |
| `LossPlotter`, `ROCPlotter`, `ConfusionMatrixPlotter` | `evaluate.plots` | Matplotlib plotters used by the metric classes (or directly). |

## Contributing

```bash
uv sync --extra torch --extra onnx --group test --group dev
uv run ruff check .
uv run ruff format .
uv run pytest
```

CI runs the same checks on Python 3.12 and 3.13, plus a `torch`-less job to
make sure the core package keeps importing without the extra installed.

## Backlog

Not currently planned, but tracked: KFold cross-validation, a GAN trainer,
and a second (non-PyTorch) backend. A second backend would follow the same
capability-interface pattern `OnnxExportable` established — implement the
capability on its own `ModelProxy` in whatever terms fit that framework,
without touching the trainer or `BaseModelProxy`.
