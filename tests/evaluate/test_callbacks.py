import numpy as np

from mlinstruct.evaluate.callbacks import accuracy, confusion_matrix_metric


def test_accuracy_matches_matching_fraction():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    assert accuracy(y_true, y_pred) == 0.75


def test_confusion_matrix_metric_matches_accuracy():
    y_true = np.array([0, 1, 1, 0, 2])
    y_pred = np.array([0, 1, 0, 0, 2])

    assert confusion_matrix_metric(y_true, y_pred) == accuracy(y_true, y_pred)
