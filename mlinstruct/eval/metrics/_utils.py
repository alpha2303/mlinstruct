import numpy as np


class MetricUtils:
    @staticmethod
    def is_valid_input_dimensions(
        truth_array: np.ndarray, pred_array: np.ndarray
    ) -> bool:
        return truth_array.ndim == 1 and np.array_equal(
            truth_array.shape, pred_array.shape
        )

    @staticmethod
    def is_valid_input_values(truth_array: np.ndarray, pred_array: np.ndarray) -> bool:
        return (
            truth_array.shape[0] > 0 
            # and len(np.setxor1d(truth_array, pred_array)) == 0
        )

class IncompatibleDimsException(ValueError):
    def __init__(self, shape_1: tuple[int], shape_2: tuple[int]):
        self.message: str = (
            f"Incompatible Dimensions: {shape_1}, {shape_2}. Size and shape of input arrays must match."
        )
        super().__init__(self.message)


class IncompatibleValuesException(ValueError):
    def __init__(
        self,
    ):
        self.message: str = (
            "Incompatible Values: Input arrays may be empty, continuous values or do not have the same unique values."
        )
        super().__init__(self.message)
