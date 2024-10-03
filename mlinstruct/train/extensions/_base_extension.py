from mlinstruct.utils import Result, Option


class BaseExtension:
    _key: str

    def __init__(self, key, **kwargs) -> None:
        self._key = key

    def get_key(self) -> str:
        return self._key

    def execute(self, **kwargs) -> Result[bool, Exception]:
        raise NotImplementedError()


class ExtensionList:
    _extensions: dict[str, BaseExtension]

    def __init__(self) -> None:
        self._extensions = dict()

    def add_extension(self, extension: BaseExtension) -> None:
        self._extensions[extension.get_key()] = extension

    def contains(self, key: str) -> bool:
        return key in self._extensions

    def get_extension(self, key) -> Option[BaseExtension]:
        return (
            Option.some(self._extensions[key]) if self.contains(key) else Option.none()
        )
