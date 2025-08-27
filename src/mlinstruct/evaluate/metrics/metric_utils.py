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
            truth_array.shape[0] > 0 and len(np.setxor1d(truth_array, pred_array)) == 0
        )
