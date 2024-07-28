from typing import Self


class Option[T]:
    def __init__(self, value: T = None):
        self._value: T | None = value
    
    @classmethod
    def some(cls, value: T) -> Self:
        return cls(value=value)
    
    @classmethod
    def none(cls) -> Self:
        return cls(None)
    
    def unwrap(self) -> T | None:
        return self._value
    
    def is_none(self) -> bool:
        return self._value is None
