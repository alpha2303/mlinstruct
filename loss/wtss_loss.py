import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn

class WTSSLoss(nn.Module):
    """
    A `torch.nn.Module` inherited loss function that computed the 
    Weighted True Skill Statistic (WTSS) metric between 
    predicted and ground truth target values.

    Setting alpha = 1.0 makes the loss computed equivalent to True Skill Statistic (TSS) metric.

    (WIP) Output compatibility with 'loss.backward()'

    Arguments:
    - alpha - `float` parameter that provides the penalty weightage 
    for false negative predictions. Default: `1.0`
    - threshold - `float` parameter that provides the probability threshold 
    used to derive the predicted targets from predicted probabilities. Default: `0.5`

    """
    def __init__(self, alpha: float=1.0, threshold: float=0.5):
        super(WTSSLoss, self).__init__()
        self.alpha = alpha
        self.threshold = threshold
    
    def _preprocess(self, inputs, targets):
        inputs = np.array([0 if y < self.threshold else 1 for y in inputs])
        return inputs, targets.cpu().numpy()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs, targets = self._preprocess(inputs, targets)
        tn, fn, fp, tp = confusion_matrix(inputs, targets).ravel()
        print(tn, fp, fn, tp)
        loss = 1. - ((2. / (self.alpha + 1.)) * ((self.alpha * (fn / (tp + fn))) + (fp / (tn + fp))))
        return torch.tensor(np.array(loss))