from typing import Iterable, Optional


class BaseDataPayload:
    """Base class for all data payloads.

    Training data, validation data, and optional test data are provided as input.

    - Training Data: Data used to train the learning model.
    - Validation Data: Data used to validate the learning model during training for hyperparameters tuning.
    - Test Data: Data used to test the model post-training, and is an optional step.

    Args:
        train_data (Iterable): The training data.
        val_data (Iterable): The validation data.
        test_data (Iterable, optional): The test data. Defaults to None.
    """

    def __init__(
        self,
        train_data: Iterable,
        val_data: Iterable,
        test_data: Optional[Iterable] = None,
    ):
        self.__train_data = train_data
        self.__val_data = val_data
        self.__test_data = test_data

    def get_train_data(self) -> Iterable:
        """Get the training data.

        Returns:
            Iterable: The training data.
        """
        return self.__train_data

    def get_val_data(self) -> Iterable:
        """Get the validation data.

        Returns:
            Iterable: The validation data.
        """
        return self.__val_data

    def get_test_data(self) -> Optional[Iterable]:
        """Get the test data.

        Returns:
            Optional[Iterable]: The test data if exists, else None.
        """
        return self.__test_data

    def has_test_data(self) -> bool:
        """Check if test data is available.

        Returns:
            bool: True if test data is available, False otherwise.
        """
        return self.__test_data is not None
