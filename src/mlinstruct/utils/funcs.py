from importlib.util import find_spec


def is_dependency_installed(package_name: str) -> bool:
    return find_spec(package_name) is not None
