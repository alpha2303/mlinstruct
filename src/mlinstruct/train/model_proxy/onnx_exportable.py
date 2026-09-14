from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class OnnxExportable(ABC):
    """Opt-in capability for model proxies whose backend can export to ONNX.

    A ModelProxy inherits from this alongside BaseModelProxy when its backend has a
    viable ONNX export path (torch.onnx, skl2onnx, tf2onnx, ...). Backends that can't
    support it simply don't inherit from it; callers check
    `isinstance(proxy, OnnxExportable)` before calling export_onnx.
    """

    @abstractmethod
    def export_onnx(self, save_path: Path, input_sample: Any, **kwargs: Any) -> Path:
        """Export the underlying model to save_path. input_sample is a representative
        single batch used for tracing/shape inference, in whatever type the backend's
        forward pass expects. **kwargs are backend-specific passthrough (e.g.
        dynamic_axes/opset_version for torch)."""
        ...
