import numpy as np


class EarlyStopper:
    """Early stopping to terminate training when validation loss stops improving.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.min_vloss: float = np.inf

    def early_stop(self, vloss: float) -> bool:
        """Check if training should be stopped early.

        Args:
            vloss (float): The current validation loss.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if vloss < self.min_vloss:
            self.min_vloss = vloss
            self.counter = 0
        elif vloss > (self.min_vloss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
