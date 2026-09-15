from typing import Any

__all__ = ["BasePlotter", "ROCPlotter", "LossPlotter", "ConfusionMatrixPlotter"]


def __getattr__(name: str) -> Any:
    if name == "BasePlotter":
        from mlinstruct.evaluate.plots.base_plotter import BasePlotter

        return BasePlotter

    if name == "ConfusionMatrixPlotter":
        from mlinstruct.evaluate.plots.cm_plotter import ConfusionMatrixPlotter

        return ConfusionMatrixPlotter

    if name == "LossPlotter":
        from mlinstruct.evaluate.plots.loss_plotter import LossPlotter

        return LossPlotter

    if name == "ROCPlotter":
        from mlinstruct.evaluate.plots.roc_plotter import ROCPlotter

        return ROCPlotter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
