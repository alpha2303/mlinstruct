from typing import Any

from mlinstruct.train.model_proxy.base_model_proxy import BaseModelProxy
from mlinstruct.train.model_proxy.onnx_exportable import OnnxExportable
from mlinstruct.utils.optional_deps import require

__all__ = ["BaseModelProxy", "OnnxExportable", "TorchModelProxy"]


def __getattr__(name: str) -> Any:
    if name == "TorchModelProxy":
        require("torch", extra="torch", symbol="TorchModelProxy")
        from mlinstruct.train.model_proxy.torch_model_proxy import TorchModelProxy

        return TorchModelProxy

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
