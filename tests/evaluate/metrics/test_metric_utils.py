import numpy as np

from mlinstruct.evaluate.metrics.metric_utils import MetricUtils


def test_is_valid_input_dimensions_true_for_matching_1d_arrays():
    assert (
        MetricUtils.is_valid_input_dimensions(np.array([1, 2, 3]), np.array([1, 2, 3]))
        is True
    )


def test_is_valid_input_dimensions_false_for_mismatched_shapes():
    assert (
        MetricUtils.is_valid_input_dimensions(np.array([1, 2, 3]), np.array([[1, 2, 3]]))
        is False
    )


def test_is_valid_input_dimensions_false_for_non_1d():
    truth = np.array([[1, 2], [3, 4]])
    pred = np.array([[1, 2], [3, 4]])

    assert MetricUtils.is_valid_input_dimensions(truth, pred) is False


def test_is_valid_input_values_true_for_matching_unique_values():
    assert (
        MetricUtils.is_valid_input_values(np.array([0, 1, 2]), np.array([2, 1, 0]))
        is True
    )


def test_is_valid_input_values_false_for_empty_array():
    assert MetricUtils.is_valid_input_values(np.array([]), np.array([])) is False


def test_is_valid_input_values_false_for_mismatched_unique_values():
    assert (
        MetricUtils.is_valid_input_values(np.array([0, 1, 2]), np.array([0, 1, 3]))
        is False
    )
