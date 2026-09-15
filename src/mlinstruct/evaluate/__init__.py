from typing import Any

__all__ = ["callbacks", "metrics", "plots"]


def __getattr__(name: str) -> Any:
    if name == "callbacks":
        from mlinstruct.evaluate import callbacks

        return callbacks

    if name == "metrics":
        from mlinstruct.evaluate import metrics

        return metrics

    if name == "plots":
        from mlinstruct.evaluate import plots

        return plots

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
