from typing import Any


def check_params(kwargs: dict, params: dict[str, type]) -> bool:
    for param, param_type in params.items():
        if param not in kwargs:
            raise ValueError(f"The parameter '{param}: {param_type}' is not provided.")
        if param_type is not Any and not isinstance(kwargs.get(param), param_type):
            raise ValueError(
                f"The value of parameter '{param}' is of type '{type(kwargs[param])}', which is not an instance of '{param_type}"
            )

    return True
