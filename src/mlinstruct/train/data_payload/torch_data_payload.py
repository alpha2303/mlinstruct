from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from mlinstruct.train.data_payload.base_data_payload import BaseDataPayload


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
        test_data: DataLoader | None = None,
    ):
        self._validate_input_data(train_data=train_data, val_data=val_data, test_data=test_data)
        super().__init__(train_data=train_data, val_data=val_data, test_data=test_data)

    def _validate_input_data(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: DataLoader | None,
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


def torch_kfold_data_payload(
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    test_data: DataLoader | None = None,
    **dataloader_kwargs: Any,
) -> TorchDataPayload:
    """Build a TorchDataPayload for one cross-validation fold.

    Wraps dataset in Subset views over the fold's train/val indices, wired into
    fresh DataLoaders. Intended to be used as a KFoldTrainer data_payload_factory,
    e.g. via functools.partial(torch_kfold_data_payload, dataset=ds, batch_size=32).

    Args:
        dataset (Dataset): The full dataset to draw the fold's samples from.
        train_idx (np.ndarray): Indices into dataset for this fold's training split.
        val_idx (np.ndarray): Indices into dataset for this fold's validation split.
        batch_size (int): Batch size for both the train and val DataLoaders.
        test_data (Optional[DataLoader]): An optional, already-built test DataLoader
            shared across all folds.
        **dataloader_kwargs (Any): Passed through to both DataLoaders.

    Returns:
        TorchDataPayload: The fold's data payload.
    """
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()), batch_size=batch_size, **dataloader_kwargs
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()), batch_size=batch_size, **dataloader_kwargs
    )
    return TorchDataPayload(train_data=train_loader, val_data=val_loader, test_data=test_data)
