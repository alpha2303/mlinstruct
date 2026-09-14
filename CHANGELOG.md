# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-09-13

### Added
- `KFoldTrainer` (`mlinstruct.train.trainer`), orchestrating K-Fold
  cross-validation: trains one independent model per fold via a
  caller-supplied `model_proxy_factory` and a delegate `BaseTrainer` per fold
  (default `DefaultTrainer`), and aggregates results into a `KFoldResult`.
  The default splitter is `sklearn.model_selection.KFold`; any
  sklearn-shaped cross-validator can be substituted. Has zero torch
  dependency — only fold *indices* are core-side; applying them to real
  tensors is pushed into the data payload factory.
- `KFoldResult` (`mlinstruct.train`), a frozen dataclass aggregating each
  fold's `TrainResult` into `mean_val_loss`, `std_val_loss`, and
  `best_fold_index`.
- `torch_kfold_data_payload` (`mlinstruct.train.data_payload`), a helper that
  turns a fold's `(train_idx, val_idx)` arrays into a `TorchDataPayload` via
  `torch.utils.data.Subset`.

### Fixed
- `tests/train/conftest.py`'s blanket module-level `pytest.importorskip("torch")`
  previously skipped collection of the entire `tests/train/` subtree
  (including already torch-free tests) whenever torch was absent. It's now
  pushed down into the individual fixtures that actually need torch, so
  torch-free orchestration code gets real no-torch CI coverage.

## [0.4.0] - 2026-09-13

### Added
- `OnnxExportable`, a pure-Python capability ABC (no framework dependency)
  that a backend-specific `ModelProxy` can multiply-inherit alongside
  `BaseModelProxy` to opt into ONNX export. Exported from
  `mlinstruct.train.model_proxy`.
- `TorchModelProxy.export_onnx(save_path, input_sample, *, dynamo=True, **kwargs)`,
  implementing `OnnxExportable` via `torch.onnx.export`. Restores the model's
  original train/eval mode after exporting; extra keyword arguments pass
  straight through to `torch.onnx.export`.
- New `mlinstruct[onnx]` extra (`onnx`, `onnxscript`), kept separate from
  `mlinstruct[torch]` since export is opt-in.
- New `torch-cuda` optional extra, declared as conflicting with `torch` via
  `tool.uv.conflicts`. Under `uv`, `mlinstruct[torch-cuda]` resolves `torch`
  from PyTorch's cu126 index automatically (`tool.uv.sources` keyed per
  extra), instead of requiring a manual `--index` override. Plain `pip`
  installs are unaffected — pip has no notion of `tool.uv.sources`, so pip
  users needing the CUDA build still pass `--index-url` themselves.

## [0.3.0] - 2026-09-13

### Added
- Opt-in automatic mixed precision (`use_amp`, `amp_dtype`) on `TorchModelProxy`.
- Opt-in gradient clipping (`max_grad_norm`) on `TorchModelProxy`.
- `TrainerCallback` hooks (`on_train_start`, `on_epoch_end`, `on_train_end`) and a
  `callbacks` argument on `DefaultTrainer`.
- `TorchModelProxy.predict` for running inference over a `DataLoader` under
  `torch.inference_mode()`.
- `evaluate.callbacks.MetricsCallback`, plus `accuracy` and `confusion_matrix_metric`
  helpers, for computing metrics against a held-out loader at every epoch end.
- `TrainResult.metrics_history`, populated from any callback exposing a
  `.history` dict.
- Optional tqdm progress bar via `DefaultTrainer(show_progress=True)` (new
  `progress` extra).
- `examples/toy_regression.py`, a runnable end-to-end example.

## [0.2.0] - 2026-09-13

### Added
- `train.utils.device.resolve_device` / `move_to_device`, and a `device` kwarg
  on `TorchModelProxy` so training can run on an accelerator when one is
  available.
- Checkpoint v2: checkpoints now carry `scheduler_state_dict` and
  `mlinstruct_version`; `TorchModelProxy.load_checkpoint` restores scheduler
  state and returns the recorded epoch.
- `DefaultTrainer.train(resume_from=...)` to resume training from a checkpoint.
- `TrainResult` is now a frozen dataclass with `best_val_loss`,
  `best_checkpoint_path`, and `stopped_early`.
- Named, collision-safe checkpoint run directories via `run_name`.
- `ruff` and `pyright` tooling; a two-job CI workflow (`test-torch` on Python
  3.12/3.13, `test-core` without the torch extra).

### Changed
- `torch` is now an optional extra (`mlinstruct[torch]`); the core package
  imports without it, and touching a torch-only symbol raises a clear
  `ImportError` instead of failing to import the whole package.
- Bumped to `numpy>=1.26,<3` and `torch>=2.6` (was `numpy<2`, `torch<2.6`) so
  the package installs cleanly on Python 3.13.
- `MetricUtils.is_valid_input_values` now validates by class range
  (non-negative, `< num_classes`) instead of requiring an exact match between
  the unique values in `y` and `y_pred` — a model that never predicts a given
  class is no longer treated as invalid input.
- `TorchModelProxy.load_weights` renamed to `load_checkpoint`; `load_weights`
  remains as a deprecated alias.

### Fixed
- `roc.py`'s AUC computation now uses `sklearn.metrics.auc` instead of
  `np.trapz`, which numpy 2 removes.

## [0.1.1] - 2026-09-13

### Fixed
- Every double-underscore attribute/method has been renamed to a single
  underscore. Name mangling meant subclasses (`DefaultTrainer`) and
  module-level helpers (in the metrics modules) were silently reading and
  writing the wrong attribute, which is why `DefaultTrainer.train`,
  `ConfusionMatrix.from_predictions`, and `ROC.from_predictions` all raised at
  runtime.
- `TorchModelProxy` was missing a concrete `get_model_name`, so it could not
  be instantiated at all.
- `CheckpointWriter` and `TorchModelProxy` each appended a `.pt` extension,
  producing `*.pt.pt` checkpoint filenames.
- The LR scheduler was stepped with the training loss instead of the
  validation loss.
- `EarlyStopper`'s plateau logic had a dead zone between `min_vloss` and
  `min_vloss + min_delta` where the counter never advanced, so a slow
  plateau never triggered early stopping.

### Removed
- Dead code (`BaseTrainer.regenerate_model_save_path`, `utils/funcs.py::check_params`)
  and every `except Exception as e: raise e` block that added nothing over
  letting the exception propagate.
