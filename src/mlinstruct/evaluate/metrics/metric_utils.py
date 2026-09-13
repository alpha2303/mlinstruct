import numpy as np


class MetricUtils:
    """Shared input validators used by the classification metric classes."""

    @staticmethod
    def is_valid_input_dimensions(truth_array: np.ndarray, pred_array: np.ndarray) -> bool:
        """True if both arrays are 1D and have matching shapes."""
        return truth_array.ndim == 1 and np.array_equal(truth_array.shape, pred_array.shape)

    @staticmethod
    def is_valid_input_values(
        truth_array: np.ndarray, pred_array: np.ndarray, num_classes: int
    ) -> bool:
        """True if both arrays are non-empty, integer-typed, and within [0, num_classes)."""
        for array in (truth_array, pred_array):
            if array.shape[0] == 0:
                return False
            if not np.issubdtype(array.dtype, np.integer):
                return False
            if np.any(array < 0) or np.any(array >= num_classes):
                return False
        return True
