from mlinstruct.utils import Result


class BaseExtension:
    key: str

    def __init__(self, key, **kwargs) -> None:
        self.key = key

    def execute(self, **kwargs) -> Result[bool, Exception]:
        raise NotImplementedError()
