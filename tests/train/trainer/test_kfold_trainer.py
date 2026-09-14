import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlinstruct.train.data_payload.base_data_payload import BaseDataPayload
from mlinstruct.train.model_proxy.base_model_proxy import BaseModelProxy
from mlinstruct.train.trainer.kfold_trainer import KFoldTrainer
from mlinstruct.utils.exception import TrainerError


class _FakeModelProxy(BaseModelProxy):
    def __init__(self) -> None:
        self._val_losses = iter([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

    def load_checkpoint(self, model_file_path):
        return 0

    def save_weights(self, epoch, save_dir_path, model_name, *args, **kwargs):
        path = save_dir_path / f"{model_name}.ckpt"
        path.write_text("checkpoint")
        return path

    def get_lr(self):
        return 0.01

    def has_scheduler(self):
        return False

    def scheduler_step(self, avg_vloss):
        pass

    def train_one_epoch(self, train_data):
        return 0.5

    def validate(self, test_data):
        return next(self._val_losses, 0.1)

    def get_model_name(self):
        return "fake-model"


class _FakeDataPayload(BaseDataPayload):
    def __init__(self, train_data, val_data, test_data=None):
        self._validate_input_data(train_data=train_data, val_data=val_data, test_data=test_data)
        super().__init__(train_data=train_data, val_data=val_data, test_data=test_data)

    def _validate_input_data(self, *args, **kwargs) -> None:
        pass


def _fake_data_payload_factory(train_idx, val_idx):
    return _FakeDataPayload(train_data=train_idx, val_data=val_idx)


def test_default_cv_is_kfold_with_five_splits(save_dir):
    trainer = KFoldTrainer(
        model_proxy_factory=_FakeModelProxy,
        data_payload_factory=_fake_data_payload_factory,
        X=np.arange(10),
        save_dir_path=save_dir,
    )

    assert isinstance(trainer._cv, KFold)
    assert trainer._cv.get_n_splits() == 5


def test_train_returns_one_fold_result_per_split(save_dir):
    trainer = KFoldTrainer(
        model_proxy_factory=_FakeModelProxy,
        data_payload_factory=_fake_data_payload_factory,
        X=np.arange(10),
        cv=KFold(n_splits=3),
        save_dir_path=save_dir,
    )

    result = trainer.train(max_epochs=1)

    assert len(result.fold_results) == 3


def test_each_fold_gets_a_freshly_constructed_model_proxy(save_dir):
    call_count = 0

    def counting_factory():
        nonlocal call_count
        call_count += 1
        return _FakeModelProxy()

    trainer = KFoldTrainer(
        model_proxy_factory=counting_factory,
        data_payload_factory=_fake_data_payload_factory,
        X=np.arange(10),
        cv=KFold(n_splits=4),
        save_dir_path=save_dir,
    )

    trainer.train(max_epochs=1)

    assert call_count == 4


def test_fold_indices_passed_to_data_payload_factory_match_cv_split(save_dir, mocker):
    cv = KFold(n_splits=3)
    X = np.arange(10)
    expected_splits = list(cv.split(X))

    factory_spy = mocker.Mock(wraps=_fake_data_payload_factory)
    trainer = KFoldTrainer(
        model_proxy_factory=_FakeModelProxy,
        data_payload_factory=factory_spy,
        X=X,
        cv=KFold(n_splits=3),
        save_dir_path=save_dir,
    )

    trainer.train(max_epochs=1)

    assert factory_spy.call_count == 3
    for call, (expected_train_idx, expected_val_idx) in zip(
        factory_spy.call_args_list, expected_splits, strict=True
    ):
        actual_train_idx, actual_val_idx = call.args
        assert np.array_equal(actual_train_idx, expected_train_idx)
        assert np.array_equal(actual_val_idx, expected_val_idx)


def test_fold_run_dirs_created_under_shared_root(save_dir):
    trainer = KFoldTrainer(
        model_proxy_factory=_FakeModelProxy,
        data_payload_factory=_fake_data_payload_factory,
        X=np.arange(10),
        cv=KFold(n_splits=2),
        save_dir_path=save_dir,
    )

    result = trainer.train(max_epochs=1)

    assert (result.model_save_path / "fold_1").exists()
    assert (result.model_save_path / "fold_2").exists()


def test_kfold_result_aggregates_mean_and_std_val_loss(save_dir):
    trainer = KFoldTrainer(
        model_proxy_factory=_FakeModelProxy,
        data_payload_factory=_fake_data_payload_factory,
        X=np.arange(10),
        cv=KFold(n_splits=3),
        save_dir_path=save_dir,
    )

    result = trainer.train(max_epochs=1)

    expected_losses = [fold.best_val_loss for fold in result.fold_results]
    assert result.mean_val_loss == np.mean(expected_losses)
    assert result.std_val_loss == np.std(expected_losses)


def test_custom_trainer_factory_is_used(save_dir, mocker):
    from mlinstruct.train.trainer.default_trainer import DefaultTrainer

    def spy_trainer_factory(model_proxy, data_payload, save_dir_path, run_name):
        return DefaultTrainer(
            model_proxy=model_proxy,
            data_payload=data_payload,
            save_dir_path=save_dir_path,
            run_name=run_name,
        )

    factory_spy = mocker.Mock(wraps=spy_trainer_factory)
    trainer = KFoldTrainer(
        model_proxy_factory=_FakeModelProxy,
        data_payload_factory=_fake_data_payload_factory,
        X=np.arange(10),
        cv=KFold(n_splits=2),
        trainer_factory=factory_spy,
        save_dir_path=save_dir,
    )

    trainer.train(max_epochs=1)

    assert factory_spy.call_count == 2
    assert factory_spy.call_args_list[0].args[3] == "fold_1"
    assert factory_spy.call_args_list[1].args[3] == "fold_2"


def test_rejects_cv_without_split_method(save_dir):
    with pytest.raises(TrainerError):
        KFoldTrainer(
            model_proxy_factory=_FakeModelProxy,
            data_payload_factory=_fake_data_payload_factory,
            X=np.arange(10),
            cv=object(),
            save_dir_path=save_dir,
        )


def test_rejects_non_positive_epochs(save_dir):
    trainer = KFoldTrainer(
        model_proxy_factory=_FakeModelProxy,
        data_payload_factory=_fake_data_payload_factory,
        X=np.arange(10),
        save_dir_path=save_dir,
    )

    with pytest.raises(ValueError):
        trainer.train(max_epochs=0)


def test_end_to_end_with_torch_model_proxy_and_data_payload(save_dir):
    torch = pytest.importorskip("torch")
    from torch import nn, optim
    from torch.utils.data import TensorDataset

    from mlinstruct.train.data_payload import torch_kfold_data_payload
    from mlinstruct.train.model_proxy import TorchModelProxy

    inputs = torch.randn(32, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    dataset = TensorDataset(inputs, targets)

    def model_proxy_factory():
        model = nn.Linear(4, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        return TorchModelProxy(model=model, optimizer=optimizer, loss_fn=nn.MSELoss())

    def data_payload_factory(train_idx, val_idx):
        return torch_kfold_data_payload(
            dataset=dataset, train_idx=train_idx, val_idx=val_idx, batch_size=8
        )

    trainer = KFoldTrainer(
        model_proxy_factory=model_proxy_factory,
        data_payload_factory=data_payload_factory,
        X=np.arange(len(dataset)),
        cv=KFold(n_splits=2),
        save_dir_path=save_dir,
    )

    result = trainer.train(max_epochs=1)

    assert len(result.fold_results) == 2
