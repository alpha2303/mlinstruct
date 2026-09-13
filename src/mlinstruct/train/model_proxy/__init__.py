from typing import Any

from .base_model_proxy import BaseModelProxy
from ...utils.optional_deps import require

__all__ = ["BaseModelProxy", "TorchModelProxy"]


def __getattr__(name: str) -> Any:
    if name == "TorchModelProxy":
        require("torch", extra="torch", symbol="TorchModelProxy")
        from .torch_model_proxy import TorchModelProxy

        return TorchModelProxy

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
