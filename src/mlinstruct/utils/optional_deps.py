from importlib.util import find_spec


def is_installed(package_name: str) -> bool:
    return find_spec(package_name) is not None


def require(name: str, extra: str, symbol: str) -> None:
    if not is_installed(name):
        raise ImportError(
            f"{symbol} requires the '{extra}' extra: pip install mlinstruct[{extra}]"
        )
