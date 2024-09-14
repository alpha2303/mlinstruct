import numpy as np
from mlinstruct.train.extensions import BaseExtension
from mlinstruct.utils import Result


class EarlyStopper(BaseExtension):
    def __init__(self, patience=2, min_delta=0.0):
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._min_vloss = np.inf
        super.__init__(key="EarlyStopper")

    def execute(self, **kwargs) -> Result[bool, Exception]:
        if "vloss" not in kwargs:
            return Result.err(
                ValueError("The parameter 'vloss: float' is not provided.")
            )
        if not isinstance(kwargs.get("vloss"), float):
            return Result.err(ValueError("The parameter 'vloss' is not a float value."))

        vloss: float = kwargs.get("vloss")
        if vloss < self._min_vloss:
            self._min_vloss = vloss
            self._counter = 0
        elif vloss > (self._min_vloss + self._min_delta):
            self._counter += 1
            if self._counter >= self._patience:
                return Result.ok(True)
        return Result.ok(False)
