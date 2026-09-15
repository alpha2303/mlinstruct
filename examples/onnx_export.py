"""ONNX export via the OnnxExportable capability interface. TorchModelProxy
implements it; callers check isinstance(proxy, OnnxExportable) rather than
assuming every backend supports export. Requires the 'onnx' extra:
uv sync --extra torch --extra onnx

Run with: uv run python examples/onnx_export.py
"""

import sys
from pathlib import Path

import torch
from torch import nn, optim

from mlinstruct.train.model_proxy import OnnxExportable, TorchModelProxy
from mlinstruct.utils.optional_deps import is_installed

# torch's dynamo-based exporter prints unicode checkmarks (e.g. "...OK") for
# each export stage; on Windows the console's default cp1252 encoding can't
# represent them, so widen stdout to utf-8 before exporting.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    if not (is_installed("onnx") and is_installed("onnxscript")):
        print(
            "Skipping: install the 'onnx' extra (pip install mlinstruct[torch,onnx]) "
            "to run this example."
        )
        return

    torch.manual_seed(0)

    model = nn.Linear(4, 1)
    proxy = TorchModelProxy(
        model=model, optimizer=optim.SGD(model.parameters(), lr=0.01), loss_fn=nn.MSELoss()
    )

    # A model proxy only supports ONNX export if its backend implements the
    # capability interface -- check before calling rather than assuming.
    if not isinstance(proxy, OnnxExportable):
        print(f"{type(proxy).__name__} does not implement OnnxExportable")
        return

    output_dir = Path("examples_output")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "toy_regression.onnx"

    input_sample = torch.randn(1, 4)
    proxy.export_onnx(onnx_path, input_sample=input_sample)

    print(f"Model exported to {onnx_path} ({onnx_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
