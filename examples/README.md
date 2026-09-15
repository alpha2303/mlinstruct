# Examples

Runnable scripts, each exercising one part of `mlinstruct` end to end. All
write their checkpoints/plots under a gitignored `examples_output/` directory.

Run any of them with `uv run python examples/<script>.py` (requires the
`torch` extra; `onnx_export.py` also requires the `onnx` extra):

```bash
uv sync --extra torch --extra onnx
```

| Script | Demonstrates |
|---|---|
| [`toy_regression.py`](toy_regression.py) | The core pipeline: `TorchDataPayload` -> `TorchModelProxy` -> `DefaultTrainer` -> `LossPlotter`. Start here. |
| [`classification_metrics.py`](classification_metrics.py) | Evaluating a classifier with `ConfusionMatrix` and `ROC` after training. |
| [`callbacks.py`](callbacks.py) | A custom `TrainerCallback` subclass and the built-in `MetricsCallback`, tracking validation accuracy every epoch. |
| [`resume_training.py`](resume_training.py) | Resuming training from a checkpoint across two separate `train()` calls. |
| [`mixed_precision_and_grad_clipping.py`](mixed_precision_and_grad_clipping.py) | `TorchModelProxy`'s `use_amp` and `max_grad_norm` options. |
| [`kfold_cross_validation.py`](kfold_cross_validation.py) | `KFoldTrainer` orchestrating one model per cross-validation fold via `torch_kfold_data_payload`. |
| [`gan_training.py`](gan_training.py) | `GANModelProxy` + `GANTrainer` training a vanilla GAN and plotting generated vs. real samples. |
| [`onnx_export.py`](onnx_export.py) | Exporting a trained model via the `OnnxExportable` capability interface. |
