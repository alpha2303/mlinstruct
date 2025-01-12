import numpy as np


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_vloss = np.inf

    def early_stop(self, vloss: float) -> bool:
        if vloss < self.min_vloss:
            self.min_vloss = vloss
            self.counter = 0
        elif vloss > (self.min_vloss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
