from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    A `torch.nn.Module` inherited wrapper class for 
    `torchvision.ops.focal_loss.sigmoid_focal_loss` method
    that can be used to compute the Focal Loss between 
    predicted and ground truth target values.

    Output is compatible with `loss.backward()` method.

    Arguments:
    - alpha - `float` multiplier parameter of the Focal Loss equation.
    - gamma - `float` exponential parameter of the Focal Loss equation.
    - reduction - `str` value indicating the reduction strategy to be used on the output. Default: "none"
        - "none": No reduction will be applied to the output.
        - "mean": The output will be averaged.
        - "sum": The output will be summed.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 0., reduction: str = "none"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)