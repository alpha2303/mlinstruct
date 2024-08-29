# Classification Metrics Exceptions


class IncompatibleDimsException(Exception):
    def __init__(self, shape_1: tuple[int], shape_2: tuple[int]):
        self.message: str = (
            f"Incompatible Dimensions: {shape_1}, {shape_2}. Size and shape of input arrays must match."
        )
        super().__init__(self.message)


class IncompatibleValuesException(Exception):
    def __init__(
        self,
    ):
        self.message: str = (
            "Incompatible Values: Input arrays may be empty or do not have the same unique values."
        )
        super().__init__(self.message)
