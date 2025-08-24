from typing import Optional

from torch.utils.data import DataLoader

from train.data_payload.base_data_payload import BaseDataPayload


class TorchDataPayload(BaseDataPayload):
    """Data payload class for PyTorch.

    This class is a specialization of the BaseDataPayload class for use with PyTorch.
    It may include additional methods or attributes specific to PyTorch data handling.

    Args:
        train_data (Iterable): The training data.
        val_data (Iterable): The validation data.
        test_data (Iterable, optional): The test data. Defaults to None.
    """

    def __init__(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: Optional[DataLoader] = None,
    ):
        self.__validate_input_data()
        super().__init__(train_data, val_data, test_data)

    def __validate_input_data(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: Optional[DataLoader],
        **kwargs,
    ) -> None:
        """
        Validate the input data loaders.

        Raises:
            TypeError: If any of the data loaders are not instances of DataLoader.
        """
        if not isinstance(train_data, DataLoader):
            raise TypeError("Expected train_data to be a DataLoader")
        if not isinstance(val_data, DataLoader):
            raise TypeError("Expected val_data to be a DataLoader")
        if test_data is not None and not isinstance(test_data, DataLoader):
            raise TypeError("Expected test_data to be a DataLoader")
