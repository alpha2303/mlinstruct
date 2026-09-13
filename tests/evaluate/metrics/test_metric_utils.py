import numpy as np

from mlinstruct.evaluate.metrics.metric_utils import MetricUtils


def test_is_valid_input_dimensions_true_for_matching_1d_arrays():
    assert MetricUtils.is_valid_input_dimensions(np.array([1, 2, 3]), np.array([1, 2, 3])) is True


def test_is_valid_input_dimensions_false_for_mismatched_shapes():
    assert (
        MetricUtils.is_valid_input_dimensions(np.array([1, 2, 3]), np.array([[1, 2, 3]])) is False
    )


def test_is_valid_input_dimensions_false_for_non_1d():
    truth = np.array([[1, 2], [3, 4]])
    pred = np.array([[1, 2], [3, 4]])

    assert MetricUtils.is_valid_input_dimensions(truth, pred) is False


def test_is_valid_input_values_true_for_values_within_range():
    assert (
        MetricUtils.is_valid_input_values(np.array([0, 1, 2]), np.array([2, 1, 0]), num_classes=3)
        is True
    )


def test_is_valid_input_values_true_when_a_class_is_never_predicted():
    assert (
        MetricUtils.is_valid_input_values(np.array([0, 1, 2]), np.array([0, 1, 1]), num_classes=3)
        is True
    )


def test_is_valid_input_values_false_for_empty_array():
    assert MetricUtils.is_valid_input_values(np.array([]), np.array([]), num_classes=3) is False


def test_is_valid_input_values_false_for_value_at_or_above_num_classes():
    assert (
        MetricUtils.is_valid_input_values(np.array([0, 1, 2]), np.array([0, 1, 3]), num_classes=3)
        is False
    )


def test_is_valid_input_values_false_for_negative_value():
    assert (
        MetricUtils.is_valid_input_values(np.array([0, -1, 2]), np.array([0, 1, 2]), num_classes=3)
        is False
    )


def test_is_valid_input_values_false_for_non_integer_dtype():
    assert (
        MetricUtils.is_valid_input_values(
            np.array([0.0, 1.0, 2.0]), np.array([0, 1, 2]), num_classes=3
        )
        is False
    )
