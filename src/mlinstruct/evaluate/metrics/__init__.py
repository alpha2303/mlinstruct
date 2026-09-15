from typing import Any

from mlinstruct.evaluate.metrics.metric_utils import MetricUtils

__all__ = ["BaseMetrics", "MetricUtils", "classification"]


def __getattr__(name: str) -> Any:
    if name == "BaseMetrics":
        from mlinstruct.evaluate.metrics.base_metrics import BaseMetrics

        return BaseMetrics

    if name == "classification":
        from mlinstruct.evaluate.metrics import classification

        return classification

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
