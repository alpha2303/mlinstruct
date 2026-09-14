from typing import Any

from mlinstruct.train.data_payload.base_data_payload import BaseDataPayload
from mlinstruct.utils.optional_deps import require

__all__ = ["BaseDataPayload", "TorchDataPayload", "torch_kfold_data_payload"]


def __getattr__(name: str) -> Any:
    if name == "TorchDataPayload":
        require("torch", extra="torch", symbol="TorchDataPayload")
        from mlinstruct.train.data_payload.torch_data_payload import TorchDataPayload

        return TorchDataPayload

    if name == "torch_kfold_data_payload":
        require("torch", extra="torch", symbol="torch_kfold_data_payload")
        from mlinstruct.train.data_payload.torch_data_payload import torch_kfold_data_payload

        return torch_kfold_data_payload

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
