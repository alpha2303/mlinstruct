# mlinstruct — Implementation Plan

Status: draft (2026-09-13). Based on the audit of `dev` @ `a586d38`.

Goal: take `mlinstruct` from "does not instantiate" to a small, tested, modern-PyTorch
training loop + evaluation toolkit that can be dropped into personal projects.

Non-goals (explicitly deferred, see Backlog): a second framework backend, KFold trainer,
GAN trainer, ONNX export, distributed training.

---

## Conventions adopted by this plan

- **Naming:** single underscore (`_attr`) for internal state everywhere. No `__name`
  identifiers remain in the package after Phase 0 (they name-mangle per class and are
  the root cause of every blocking bug).
- **Backends are install-time extras.** `pip install mlinstruct` gives the framework-agnostic
  core (`Base*` ABCs, `evaluate`, `TrainResult`, `EarlyStopper`, `CheckpointWriter`);
  `pip install mlinstruct[torch]` adds `TorchModelProxy` / `TorchDataPayload`. Only the
  torch extra is in scope. Consequences that the plan enforces:
  - `import mlinstruct` and everything under `mlinstruct.evaluate` must work with torch
    absent (today it raises `ImportError` — see 1.5).
  - Torch symbols are exposed via lazy module `__getattr__` (PEP 562); accessing them without
    torch raises `ImportError("TorchModelProxy requires the 'torch' extra: pip install mlinstruct[torch]")`.
  - `torch` is only ever imported inside `*/torch_*.py` modules and inside functions.
  - CI runs one leg **without** the extra to keep this honest.
- **Tests:** pytest-style functions (existing `unittest.TestCase` files are kept and made
  green, not rewritten). `tests/` mirrors `src/mlinstruct/` (`tests/evaluation/` becomes
  `tests/evaluate/`). Shared fixtures live in `tests/conftest.py`. `pytest-cov` is added;
  target >= 85 % line coverage on `mlinstruct.train`, >= 75 % overall, enforced in CI from Phase 1.
- **One PR per phase**, small commits per task below (`[fix]`, `[feat]`, `[test]`,
  `[chore]` prefixes as in existing history). Each phase ends with the full suite green.
- Version bumps: Phase 0 -> `0.1.1`, Phase 1 -> `0.2.0`, Phase 2 -> `0.3.0`.

---

## Phase 0 — Make it run  (~1/2 day)

Exit criteria: `uv run pytest` is fully green; a smoke test trains `nn.Linear` for two
epochs end-to-end and finds a checkpoint on disk.

### 0.1 Test scaffolding (do first — everything else is verified against it)
- `tests/conftest.py`
  - `matplotlib.use("Agg")` at import so plot tests never open a window.
  - Fixtures: `tiny_model` (`nn.Linear(4, 1)`), `tiny_loaders` (`TensorDataset` of 64x4
    regression samples, `batch_size=16`, returns train/val/test `DataLoader`s),
    `proxy` (`TorchModelProxy` with SGD + `MSELoss`), `payload` (`TorchDataPayload`),
    `save_dir` (wraps `tmp_path`).
  - `torch.manual_seed(0)` autouse fixture for determinism.
- `tests/train/conftest.py`: `torch = pytest.importorskip("torch")` at module top so every
  torch-dependent test is skipped (not errored) in a no-torch environment. Torch fixtures
  (`tiny_model`, `tiny_loaders`, `proxy`, `payload`) live here, not in the root conftest.
- Add `pytest-cov` to the `test` dependency group.
- Move `tests/evaluation/` to `tests/evaluate/` to mirror `src`.
- Files: `tests/conftest.py`, `tests/train/conftest.py`, `pyproject.toml`.

### 0.2 Name-mangling sweep
Replace every `__x` attribute/method/function with `_x`. Affected identifiers:

| File | Identifiers |
|---|---|
| `train/trainer/base_trainer.py` | `__model_proxy`, `__data_payload`, `__checkpoint_writer`, `__root_save_dir_path`, `__early_stopper`, `__model_save_dir_path` |
| `train/trainer/default_trainer.py` | `__logger`, `__validate_trainer_attrs` + all `self.__*` reads |
| `train/data_payload/base_data_payload.py` | `__train_data`, `__val_data`, `__test_data`, abstract `__validate_input_data` |
| `train/data_payload/torch_data_payload.py` | `__validate_input_data` |
| `train/model_proxy/torch_model_proxy.py` | `__model`, `__optimizer`, `__loss_fn`, `__scheduler` |
| `train/utils/checkpoint_writer.py` | `__root_save_dir_path`, `__model_save_dir_path` |
| `evaluate/metrics/classification/confusion_matrix.py` | module fn `__compute_confusion_matrix`, `__confusion_matrix`, `__class_labels` |
| `evaluate/metrics/classification/roc.py` | module fns `__compute_roc_curve`, `__compute_auc`, `__fpr`, `__tpr`, `__auc` |
| `evaluate/plots/*_plotter.py` | all `self.__*` config attrs, `__get_text_color` |

- Tests: the existing `test_construct_cm_from_predictions` turns green; new
  `tests/evaluate/metrics/classification/test_roc.py::test_from_predictions_runs`.

### 0.3 `TorchModelProxy.get_model_name`
- Add `model_name: str | None = None` ctor kwarg; default `type(model).__name__`.
- Tests: `test_torch_model_proxy.py::test_default_model_name`, `::test_custom_model_name`.

### 0.4 Checkpoint filename `.pt.pt`
- `CheckpointWriter.create_checkpoint` passes a stem (`model_epoch_{e}_vloss_{v:.4f}`);
  the proxy owns the extension. `create_checkpoint` returns the written `Path`.
- Tests: `test_checkpoint_writer.py::test_checkpoint_filename_has_single_extension`,
  `::test_create_checkpoint_returns_existing_path`.

### 0.5 Scheduler receives validation loss
- `default_trainer.py`: `scheduler_step(avg_vloss=avg_vloss)`.
- Tests: `test_default_trainer.py::test_scheduler_step_receives_val_loss` — use
  `pytest-mock` to spy on `proxy.scheduler_step` and assert the kwarg equals the value
  returned by `proxy.validate`.

### 0.6 `EarlyStopper` plateau logic
- Rewrite: `if vloss < self.min_vloss - self.min_delta: reset; else: counter += 1`;
  return `counter >= patience`.
- Tests: `test_early_stopper.py` — improving sequence never stops; flat plateau stops
  after `patience` epochs; improvement resets counter; `min_delta` boundary case.

### 0.7 Dead code / hygiene
- Remove `BaseTrainer.regenerate_model_save_path` and its `_TIMESTAMP_FORMAT` copy.
- Remove `utils/funcs.py::check_params` (unused).
- Remove every `try: ... except Exception as e: raise e` block.
- `DefaultTrainer.__init__(..., logger: logging.Logger | None = None)`, resolved inside.
- `ConfusionMatrix.as_ndarray` return type `np.ndarray` (never `None`).
- `CheckpointWriter.__init__` no longer creates directories; `train()` calls
  `regenerate_model_save_path()` once.
- Tests: `test_default_trainer.py::test_default_logger_is_module_logger`.

### 0.8 Smoke + unit tests for the training path (new)
- `tests/train/trainer/test_default_trainer.py`
  - `test_train_returns_result_with_correct_epochs`
  - `test_train_loss_lists_have_len_epochs`
  - `test_train_writes_best_checkpoint` (assert >= 1 `.pt` file and that it loads)
  - `test_train_rejects_non_positive_epochs`
  - `test_early_stop_breaks_loop` (mock `validate` to return a constant, stops at `patience`)
  - `test_test_set_evaluated_when_present` (spy `validate` called with test loader)
  - `test_constructor_rejects_wrong_types` (raises `TrainerError`)
- `tests/train/model_proxy/test_torch_model_proxy.py`
  - `test_train_one_epoch_reduces_loss` (two epochs on a linearly-separable toy set)
  - `test_train_one_epoch_rejects_non_dataloader`
  - `test_validate_does_not_update_weights` (state_dict equality before/after)
  - `test_save_then_load_roundtrip` (weights + optimizer state equal)
  - `test_save_weights_rejects_missing_dir`
  - `test_has_scheduler_false_by_default`, `test_scheduler_step_plateau_vs_step`
  - `test_summary_returns_model_statistics`
- `tests/train/data_payload/test_torch_data_payload.py`
  - valid construction; `has_test_data` true/false; `TypeError` for each non-DataLoader slot.
- `tests/train/test_train_result.py` — field round-trip.
- `tests/evaluate/plots/test_loss_plotter.py`, `test_roc_plotter.py` — plot on `Agg`
  axes with/without validation/AUC; assert title, labels, line count, legend presence.
- `tests/evaluate/metrics/test_metric_utils.py` — dimension and value validators.

---

## Phase 1 — Modern PyTorch minimum  (~1 day)

Exit criteria: installs and passes on Python 3.12 **and** 3.13 with numpy 2, both with
and without the `torch` extra (torch >= 2.6); trains on GPU when present; can resume from
a checkpoint; CI green.

### 1.1 Device placement
- New `train/utils/device.py::resolve_device(device: str | torch.device | None) -> torch.device`
  - explicit arg wins; else `torch.accelerator.current_accelerator()` if available
    (torch >= 2.6) else `cpu`.
- New `train/utils/device.py::move_to_device(obj, device, non_blocking=True)` — recursive
  over `Tensor`, `tuple`, `list`, `dict`; anything else returned unchanged.
- `TorchModelProxy.__init__(..., device=None)` sets `self._device = resolve_device(device)`;
  `model.to(self._device)` in ctor; expose `.device` property.
- `train_one_epoch` / `validate`: `inputs, targets = move_to_device(batch, self._device)`.
  Batch contract stays "a 2-tuple `(inputs, targets)`"; documented in the class docstring.
- Tests (`test_torch_model_proxy.py`):
  - `test_resolve_device_explicit_string`, `_explicit_device`, `_default_is_cpu_when_no_accelerator` (monkeypatch)
  - `test_move_to_device_nested_structures`
  - `test_model_is_on_requested_device`
  - `@pytest.mark.skipif(not torch.cuda.is_available())` `test_train_one_epoch_on_cuda`
  - `test_train_with_cpu_loader_and_gpu_model_does_not_raise` (skipif no cuda)

### 1.2 Checkpoint v2 + resume
- Checkpoint dict: `{"epoch", "model_state_dict", "optimizer_state_dict",
  "scheduler_state_dict" | None, "loss", "mlinstruct_version"}`.
- `TorchModelProxy.load_weights(path)` renamed to `load_checkpoint(path) -> int` (returns
  epoch); uses `torch.load(path, map_location=self._device, weights_only=True)`;
  restores scheduler if present; **does not** call `.eval()`. Keep `load_weights` as a
  thin deprecated alias for one release.
- `DefaultTrainer.train(max_epochs, resume_from: Path | None = None)`: if given, loads
  the checkpoint and starts at `epoch + 1`; `max_epochs` remains the absolute upper bound.
- Tests:
  - `test_checkpoint_contains_scheduler_state_when_present` / `_none_when_absent`
  - `test_load_checkpoint_returns_epoch_and_keeps_train_mode`
  - `test_load_checkpoint_uses_weights_only` (mock `torch.load`, assert kwarg)
  - `test_load_checkpoint_map_location_matches_device`
  - `test_default_trainer.py::test_resume_continues_from_saved_epoch` (train 2, resume,
    train to 4: `TrainResult.epochs == 4`, loss lists len 2 for the second run)

### 1.3 `TrainResult` as a dataclass with checkpoint info
- `@dataclass(frozen=True) TrainResult`: existing fields + `best_val_loss: float`,
  `best_checkpoint_path: Path | None`, `stopped_early: bool`.
- `DefaultTrainer` fills them from `CheckpointWriter.create_checkpoint`'s return value.
- Tests: `test_train_result_reports_best_checkpoint_path_that_exists`,
  `test_stopped_early_flag_set_when_early_stopper_triggers`.

### 1.4 Unique run directories
- Timestamp format `%Y%m%d_%H%M%S`; optional `run_name: str | None` on the trainer
  (`<root>/<run_name or timestamp>`); on collision append `_1`, `_2`, ...
- Tests: `test_checkpoint_writer.py::test_run_dir_uses_run_name`,
  `::test_run_dir_collision_gets_suffix` (pre-create the dir).

### 1.5 Dependencies, optional torch extra & packaging
- `pyproject.toml`
  - `dependencies`: `numpy>=1.26,<3`, `matplotlib>=3.8,<4`, `scikit-learn>=1.5,<2`.
    Remove `markupsafe`.
  - `[project.optional-dependencies].torch = ["torch>=2.6", "torchinfo>=1.8,<2"]`.
    Remove `torchvision` (unused).
  - Keep `requires-python = ">=3.12,<4"`. Keep the CPU index for `uv` dev installs
    (document `--index pytorch-cu126` for GPU).
  - Regenerate `uv.lock`; confirm `uv sync --python 3.13 --extra torch` resolves numpy 2.x.
- Make the core importable without torch (currently `import mlinstruct` raises):
  - New `utils/optional_deps.py`:
    `is_installed(name) -> bool` (replaces `funcs.is_dependency_installed`) and
    `require(name: str, extra: str, symbol: str) -> None` that raises the standard
    `ImportError` message above.
  - `train/model_proxy/__init__.py` and `train/data_payload/__init__.py`: import only the
    `Base*` class eagerly; add a module-level `__getattr__(name)` that lazily imports
    `TorchModelProxy` / `TorchDataPayload` on first access and otherwise raises
    `AttributeError`. Keep them in `__all__` so IDEs/docs still see them.
  - `torch_model_proxy.py` / `torch_data_payload.py`: drop the top-of-file
    `is_dependency_installed` guard (the lazy `__getattr__` is the guard now); the plain
    `import torch` failing is acceptable inside these modules.
  - `torchinfo` is imported inside `TorchModelProxy.summary()` only.
- Move `src/py.typed` to `src/mlinstruct/py.typed`.
- `roc.py`: replace `np.trapz` with `sklearn.metrics.auc`.
- `TorchModelProxy` type hints: `loss_fn: nn.Module`, `scheduler: LRScheduler | None`.
- Tests:
  - `tests/test_package.py::test_import_top_level`, `::test_version_matches_pyproject`,
    `::test_py_typed_is_packaged` (uses `importlib.resources`)
  - `tests/test_package.py::test_core_imports_without_torch` — runs a subprocess with
    `sys.modules["torch"] = None` and asserts `import mlinstruct`,
    `import mlinstruct.evaluate`, `import mlinstruct.train` all succeed.
  - `tests/test_package.py::test_torch_symbols_raise_helpful_error_without_torch` — same
    subprocess technique; `from mlinstruct.train.model_proxy import TorchModelProxy` raises
    `ImportError` whose message contains `mlinstruct[torch]`.
  - `tests/test_package.py::test_lazy_getattr_unknown_name_raises_attribute_error`.
  - `tests/train/model_proxy/test_torch_model_proxy.py::test_lazy_import_returns_same_class_twice`.
  - `test_roc.py::test_auc_matches_sklearn_roc_auc_score`
  - CI (below) covers the numpy-2 / 3.13 and no-torch requirements.

### 1.6 Metric input validation
- `ConfusionMatrix.from_predictions(y, y_pred, num_classes: int | None = None, class_labels=None)`;
  `num_classes` defaults to `max(y.max(), y_pred.max()) + 1`.
- `MetricUtils.is_valid_input_values` checks non-empty, integer dtype, non-negative,
  `< num_classes`. Drop the `setxor1d` requirement (a model that never predicts a class
  is valid input).
- Vectorise `_compute_confusion_matrix` with `np.add.at` / `np.bincount`.
- Update the existing `TestConfusionMatrix` expectations accordingly
  (`test_cm_construct_from_predictions_invalid_values_fail` now uses a negative label).
- Tests: `test_missing_predicted_class_is_valid`, `test_num_classes_override_pads_matrix`,
  `test_non_contiguous_labels`, `test_matches_sklearn_confusion_matrix` (parametrised
  random inputs).

### 1.7 Tooling & CI
- `pyproject.toml`: `[tool.ruff]` (line-length 100, `select = ["E","F","I","UP","B"]`),
  `[tool.ruff.format]`; `[tool.pytest.ini_options]` add `addopts = "--cov=mlinstruct --cov-report=term-missing"`,
  `testpaths = ["tests"]`; `[tool.pyright]` `typeCheckingMode = "basic"`, run in CI as
  non-blocking initially.
- `.github/workflows/ci.yml`: `ubuntu-latest`, `astral-sh/setup-uv`, two jobs:
  - `test-torch`: matrix `python: [3.12, 3.13]`; `uv sync --extra torch --group test --group dev`,
    `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest --cov-fail-under=75`.
  - `test-core` (no extra): `python: 3.13`; `uv sync --group test`; `uv run pytest` —
    torch tests skip via `importorskip`, and `tests/test_package.py` asserts the no-torch
    contract. Coverage gate not applied to this job.
- Add `dev` group: `ruff`, `pyright`.
- Optional: `.pre-commit-config.yaml` with ruff hooks.

---

## Phase 2 — Worth having  (~1-2 days, can be split into independent PRs)

Exit criteria: opt-in AMP and gradient clipping work and are tested on CPU (and CUDA
when present); training can emit per-epoch metrics from `mlinstruct.evaluate`;
README has a copy-pasteable quickstart.

### 2.1 Mixed precision (opt-in)
- `TorchModelProxy(..., use_amp: bool = False, amp_dtype: torch.dtype | None = None)`.
  Default dtype: `bfloat16` if `device.type == "cuda"` and `torch.cuda.is_bf16_supported()`,
  else `float16`; on CPU autocast uses `bfloat16`.
- `torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)`.
- `train_one_epoch`: wrap forward + loss in `torch.autocast(device_type, dtype, enabled=use_amp)`;
  `scaler.scale(loss).backward(); scaler.step(opt); scaler.update()`.
- Add `"scaler_state_dict"` to the checkpoint; restore in `load_checkpoint`.
- Tests: `test_amp_disabled_by_default_has_no_scaler_effect`, `test_amp_cpu_bf16_trains`,
  `skipif` `test_amp_cuda_fp16_uses_scaler` (spy on `scaler.step`), `test_checkpoint_roundtrips_scaler_state`.

### 2.2 Gradient clipping (opt-in)
- `TorchModelProxy(..., max_grad_norm: float | None = None)`; when set, `scaler.unscale_(opt)`
  then `clip_grad_norm_(model.parameters(), max_grad_norm)` before `step`.
- Tests: `test_grad_norm_is_clipped` (huge lr + check that the post-clip total norm <= bound,
  computed from `p.grad` after a step via a wrapped optimizer), `test_no_clipping_when_none`.

### 2.3 Callbacks + evaluation-during-training
- New `train/callbacks.py`:
  ```python
  class TrainerCallback:
      def on_train_start(self, trainer): ...
      def on_epoch_end(self, trainer, epoch, train_loss, val_loss): ...
      def on_train_end(self, trainer, result): ...
  ```
  `DefaultTrainer(..., callbacks: Sequence[TrainerCallback] = ())`; trainer invokes them
  at the three points. `EarlyStopper` and checkpointing stay built-in (not callbacks) to
  keep the default loop readable.
- `TorchModelProxy.predict(data: DataLoader) -> tuple[np.ndarray, np.ndarray]` returns
  concatenated `(y_true, y_pred_raw)` on CPU under `torch.inference_mode()`.
- `evaluate/callbacks.py::MetricsCallback(metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]], loader)`
  computes each metric on `predict(loader)` at epoch end; history exposed as
  `callback.history: dict[str, list[float]]` and copied into `TrainResult.metrics_history`.
  Ships with helpers `accuracy`, `confusion_matrix_metric` (wraps `ConfusionMatrix`).
- Tests: `test_callbacks_invoked_in_order` (recording callback), `test_predict_shapes_and_no_grad`,
  `test_metrics_callback_history_length_equals_epochs`, `test_metrics_history_in_train_result`.

### 2.4 Progress reporting
- `DefaultTrainer(..., show_progress: bool = False)`; uses `tqdm` if importable
  (`progress` optional extra), silently no-ops otherwise.
- Tests: `test_show_progress_without_tqdm_does_not_raise` (monkeypatch import failure).

### 2.5 Documentation & release
- README rewrite: what it is (a small, readable training loop, not a Lightning
  competitor), install (CPU/GPU), 20-line quickstart (`TorchDataPayload` -> `TorchModelProxy`
  -> `DefaultTrainer` -> `LossPlotter` + `ConfusionMatrix`), resume example, AMP example,
  callback example, API table, contributing (uv, ruff, pytest).
- `CHANGELOG.md` (Keep-a-Changelog); `docs/` keeps this plan; `examples/toy_regression.py`
  is a runnable toy script that CI executes as a smoke test.
- Docstring pass on all public classes; bump to `0.3.0`; tag release.

---

## Backlog (out of scope for these phases)

- KFold cross-validation trainer (build on `TrainerCallback` + `TorchDataPayload` factories).
- GAN trainer (needs a multi-optimizer `ModelProxy`; design after 2.3 lands).
- ONNX export (`torch.onnx.export` wrapper on the proxy).
- Pydantic validation (revisit only if the config surface grows; dataclasses suffice now).
- Second framework backend.

---

## Decisions (resolved 2026-09-13)

1. Torch stays an **optional extra** (`mlinstruct[torch]`); the core must import without
   it. Only the torch backend is in scope. See Conventions and 1.5.
2. Batch contract stays `(inputs, targets)` 2-tuple; a user-supplied `unpack_batch` hook
   can be added in 2.3 if needed.
3. `load_weights` is renamed to `load_checkpoint` with a deprecated alias for one release.
4. Type checker: pyright `basic`.
