from typing import Tuple

"""
Model Training Exceptions
"""


class TrainerError(Exception):
    """Exception raised for errors in the trainer."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ModelProxyError(Exception):
    """Exception raised for errors in the model proxy."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


"""
Model Evaluation Exceptions
"""


class IncompatibleDimsException(ValueError):
    """Exception raised for incompatible input dimensions for ndarrays."""

    def __init__(self, shape_1: Tuple[int, ...], shape_2: Tuple[int, ...]) -> None:
        self.message: str = f"Incompatible Dimensions: {shape_1}, {shape_2}. Size and shape of input arrays must match."
        super().__init__(self.message)


class IncompatibleValuesException(ValueError):
    """Exception raised for incompatible input values for ndarrays."""

    def __init__(self) -> None:
        self.message: str = "Incompatible Values: Input arrays may be empty, continuous values or do not have the same unique values."
        super().__init__(self.message)
