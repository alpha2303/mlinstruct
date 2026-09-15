from typing import Any

__all__ = ["ConfusionMatrix", "ROC"]


def __getattr__(name: str) -> Any:
    if name == "ConfusionMatrix":
        from mlinstruct.evaluate.metrics.classification.confusion_matrix import ConfusionMatrix

        return ConfusionMatrix

    if name == "ROC":
        from mlinstruct.evaluate.metrics.classification.roc import ROC

        return ROC

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
