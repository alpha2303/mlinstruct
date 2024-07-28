from ivy.evaluation.metrics import ConfusionMatrix
import numpy as np

from unittest import TestCase

class TestConfusionMatrix(TestCase):
    def setUp(self) -> None:
        self.class_labels: list[str] = ['Car', 'Bike', 'Scooter']
        self.y: np.ndarray = np.array(
            [2, 2, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 1, 0, 2, 2, 1, 0, 0, 0, 0, 2, 1, 1, 2, 2, 2, 1, 0, 1, 0, 0, 0]
        )
        self.y_pred: np.ndarray = np.array(
            [2, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 2])
        self.confusion_matrix: np.ndarray = np.array([[8, 6, 8], [4, 4, 5], [4, 4, 7]])
    
    def test_cm_as_array(self)-> None:
        cm = ConfusionMatrix(self.confusion_matrix, self.class_labels)
        self.assertFalse(cm.as_array().is_none()) 
        self.assertTrue(np.array_equal(cm.as_array().unwrap(), self.confusion_matrix))
    
    def test_cm_construct_from_predictions(self) -> None:
        cm_result = ConfusionMatrix.from_predictions(self.y, self.y_pred, self.class_labels).unwrap()
        self.assertFalse(isinstance(cm_result, Exception))
        self.assertTrue(np.array_equal(cm_result.as_array().unwrap(), self.confusion_matrix))


