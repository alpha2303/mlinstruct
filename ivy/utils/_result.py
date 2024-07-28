from typing import Self

class Result[T, E: Exception]:
    def __init__(self, value: T = None, err: E = None):
        self._value = value
        self._err = err
    
    @classmethod
    def ok(cls, value: T) -> Self:
        return cls(value=value)
    
    @classmethod
    def err(cls, err: E) -> Self:
        return cls(err=err)
    
    def is_err(self) -> bool:
        return self._err is not None
    
    def unwrap(self) -> T | E:
        if self.is_err():
            return self._err
        return self._value