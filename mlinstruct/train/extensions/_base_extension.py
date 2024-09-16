from mlinstruct.utils import Result


class BaseExtension:
    _key: str

    def __init__(self, key, **kwargs) -> None:
        self._key = key
    
    def getKey(self) -> str:
        return self._key

    def execute(self, **kwargs) -> Result[bool, Exception]:
        raise NotImplementedError()
