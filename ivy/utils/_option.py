class Option[T]:
    def __init__(self, value: T = None):
        self._value: T | None = value
    
    def unwrap(self) -> T | None:
        return self._value
    
    def is_none(self) -> bool:
        return self._value == None
