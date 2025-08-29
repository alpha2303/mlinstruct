from typing import Any
from importlib.util import find_spec


def check_params(kwargs: dict, params: dict[str, type]) -> bool:
    for param, param_type in params.items():
        if param not in kwargs:
            raise ValueError(f"The parameter '{param}: {param_type}' is not provided.")
        if param_type is not Any and not isinstance(kwargs.get(param), param_type):
            raise ValueError(
                f"The value of parameter '{param}' is of type '{type(kwargs[param])}', which is not an instance of '{param_type}"
            )

    return True


def is_dependency_installed(package_name: str) -> bool:
    """
    Check if a Python package is installed.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    return find_spec(package_name) is not None
