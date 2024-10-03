from ._result import Result
from typing import Any


def check_params(kwargs: dict, params: dict[str, type]) -> Result[bool, Exception]:
    for param, param_type in params.items():
        if param not in kwargs:
            return Result.err(ValueError(f"The parameter '{param}: {param_type}' is not provided."))
        if param_type is not Any and not isinstance(kwargs.get(param), param_type):
            return Result.err(ValueError(f"The value of parameter '{param}' is not an instance of '{param_type}"))
    
    return Result.ok(True)
