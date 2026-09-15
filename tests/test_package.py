import subprocess
import sys
import tomllib
from importlib import resources
from pathlib import Path

import pytest

import mlinstruct


def test_import_top_level():
    import mlinstruct.evaluate  # noqa: F401
    import mlinstruct.train  # noqa: F401
    import mlinstruct.utils  # noqa: F401


def test_version_matches_pyproject():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject_path, "rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert mlinstruct.__version__ == pyproject["project"]["version"]


def test_py_typed_is_packaged():
    assert resources.files("mlinstruct").joinpath("py.typed").is_file()


# scipy's array-api-compat shim indexes sys.modules["torch"] without a presence
# check, so it must be imported (and its torch-detection code run once, for real)
# before we poison sys.modules; otherwise scipy crashes on the sentinel itself.
_BLOCK_TORCH = "import sklearn.metrics\nimport sys\nsys.modules['torch'] = None\n"


def _run_without_torch(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH + code],
        capture_output=True,
        text=True,
    )


def test_core_imports_without_torch():
    result = _run_without_torch(
        "import mlinstruct\nimport mlinstruct.evaluate\nimport mlinstruct.train\n"
    )

    assert result.returncode == 0, result.stderr


def test_torch_symbols_raise_helpful_error_without_torch():
    result = _run_without_torch("from mlinstruct.train.model_proxy import TorchModelProxy\n")

    assert result.returncode != 0
    assert "mlinstruct[torch]" in result.stderr


def test_lazy_getattr_unknown_name_raises_attribute_error():
    import mlinstruct.train.model_proxy as model_proxy_module

    with pytest.raises(AttributeError):
        _ = model_proxy_module.DoesNotExist


def test_data_payload_lazy_getattr_unknown_name_raises_attribute_error():
    import mlinstruct.train.data_payload as data_payload_module

    with pytest.raises(AttributeError):
        _ = data_payload_module.DoesNotExist


def test_trainer_lazy_getattr_unknown_name_raises_attribute_error():
    import mlinstruct.train.trainer as trainer_module

    with pytest.raises(AttributeError):
        _ = trainer_module.DoesNotExist


def test_kfold_trainer_importable_without_torch():
    result = _run_without_torch(
        "from mlinstruct.train.trainer import KFoldTrainer\n"
        "from mlinstruct.train import KFoldResult\n"
    )

    assert result.returncode == 0, result.stderr


def test_gan_model_proxy_without_torch_extra_raises_import_error():
    result = _run_without_torch("from mlinstruct.train.model_proxy import GANModelProxy\n")

    assert result.returncode != 0
    assert "mlinstruct[torch]" in result.stderr


def test_gan_trainer_without_torch_extra_raises_import_error():
    result = _run_without_torch("from mlinstruct.train.trainer import GANTrainer\n")

    assert result.returncode != 0
    assert "mlinstruct[torch]" in result.stderr


def test_evaluate_lazy_submodules_resolve():
    import mlinstruct.evaluate as evaluate_module
    from mlinstruct.evaluate import callbacks, metrics, plots

    assert evaluate_module.callbacks is callbacks
    assert evaluate_module.metrics is metrics
    assert evaluate_module.plots is plots


def test_evaluate_lazy_getattr_unknown_name_raises_attribute_error():
    import mlinstruct.evaluate as evaluate_module

    with pytest.raises(AttributeError):
        _ = evaluate_module.DoesNotExist


def test_evaluate_metrics_lazy_symbols_resolve():
    import mlinstruct.evaluate.metrics as metrics_module
    from mlinstruct.evaluate.metrics import classification
    from mlinstruct.evaluate.metrics.base_metrics import BaseMetrics

    assert metrics_module.BaseMetrics is BaseMetrics
    assert metrics_module.classification is classification


def test_evaluate_metrics_lazy_getattr_unknown_name_raises_attribute_error():
    import mlinstruct.evaluate.metrics as metrics_module

    with pytest.raises(AttributeError):
        _ = metrics_module.DoesNotExist


def test_evaluate_metrics_classification_lazy_getattr_unknown_name_raises_attribute_error():
    import mlinstruct.evaluate.metrics.classification as classification_module

    with pytest.raises(AttributeError):
        _ = classification_module.DoesNotExist


def test_evaluate_plots_lazy_symbols_resolve():
    import mlinstruct.evaluate.plots as plots_module
    from mlinstruct.evaluate.plots.base_plotter import BasePlotter
    from mlinstruct.evaluate.plots.cm_plotter import ConfusionMatrixPlotter
    from mlinstruct.evaluate.plots.loss_plotter import LossPlotter
    from mlinstruct.evaluate.plots.roc_plotter import ROCPlotter

    assert plots_module.BasePlotter is BasePlotter
    assert plots_module.ConfusionMatrixPlotter is ConfusionMatrixPlotter
    assert plots_module.LossPlotter is LossPlotter
    assert plots_module.ROCPlotter is ROCPlotter


def test_evaluate_plots_lazy_getattr_unknown_name_raises_attribute_error():
    import mlinstruct.evaluate.plots as plots_module

    with pytest.raises(AttributeError):
        _ = plots_module.DoesNotExist
