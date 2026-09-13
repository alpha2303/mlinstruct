from typing import Any

from ...utils.optional_deps import require
from .base_data_payload import BaseDataPayload

__all__ = ["BaseDataPayload", "TorchDataPayload"]


def __getattr__(name: str) -> Any:
    if name == "TorchDataPayload":
        require("torch", extra="torch", symbol="TorchDataPayload")
        from .torch_data_payload import TorchDataPayload

        return TorchDataPayload

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
