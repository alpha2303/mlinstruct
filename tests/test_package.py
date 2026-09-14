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


def test_kfold_trainer_importable_without_torch():
    result = _run_without_torch(
        "from mlinstruct.train.trainer import KFoldTrainer\n"
        "from mlinstruct.train import KFoldResult\n"
    )

    assert result.returncode == 0, result.stderr
