import pytest

torch = pytest.importorskip("torch")

from mlinstruct.train.callbacks import TrainerCallback  # noqa: E402
from mlinstruct.train.trainer import GANTrainer  # noqa: E402


class RecordingCallback(TrainerCallback):
    def __init__(self) -> None:
        self.events: list = []

    def on_train_start(self, trainer) -> None:
        self.events.append(("start",))

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss) -> None:
        self.events.append(("epoch_end", epoch, train_loss, val_loss))

    def on_train_end(self, trainer, result) -> None:
        self.events.append(("end",))


def test_train_returns_result_with_correct_epochs(gan_proxy, real_samples_loader, save_dir):
    trainer = GANTrainer(
        model_proxy=gan_proxy, train_data=real_samples_loader, save_dir_path=save_dir
    )

    result = trainer.train(max_epochs=2)

    assert result.epochs == 2


def test_g_loss_and_d_loss_lists_have_len_epochs(gan_proxy, real_samples_loader, save_dir):
    trainer = GANTrainer(
        model_proxy=gan_proxy, train_data=real_samples_loader, save_dir_path=save_dir
    )

    result = trainer.train(max_epochs=2)

    assert len(result.g_loss_list) == 2
    assert len(result.d_loss_list) == 2


def test_checkpoint_written_at_configured_interval(gan_proxy, real_samples_loader, save_dir):
    trainer = GANTrainer(
        model_proxy=gan_proxy,
        train_data=real_samples_loader,
        save_dir_path=save_dir,
        checkpoint_interval=2,
    )

    result = trainer.train(max_epochs=4)

    checkpoints = sorted(result.model_save_path.glob("*.pt"))
    assert len(checkpoints) == 2
    assert "epoch_2" in checkpoints[0].name
    assert "epoch_4" in checkpoints[1].name


def test_final_epoch_always_checkpointed_even_off_interval(
    gan_proxy, real_samples_loader, save_dir
):
    trainer = GANTrainer(
        model_proxy=gan_proxy,
        train_data=real_samples_loader,
        save_dir_path=save_dir,
        checkpoint_interval=2,
    )

    result = trainer.train(max_epochs=3)

    checkpoints = sorted(result.model_save_path.glob("*.pt"))
    assert len(checkpoints) == 2
    assert "epoch_2" in checkpoints[0].name
    assert "epoch_3" in checkpoints[1].name


def test_callbacks_receive_generator_loss_as_train_loss_and_discriminator_loss_as_val_loss(
    gan_proxy, real_samples_loader, save_dir
):
    callback = RecordingCallback()
    trainer = GANTrainer(
        model_proxy=gan_proxy,
        train_data=real_samples_loader,
        save_dir_path=save_dir,
        callbacks=[callback],
    )

    result = trainer.train(max_epochs=2)

    epoch_events = [event for event in callback.events if event[0] == "epoch_end"]
    assert [event[1] for event in epoch_events] == [1, 2]
    assert [event[2] for event in epoch_events] == result.g_loss_list
    assert [event[3] for event in epoch_events] == result.d_loss_list
    assert callback.events[0] == ("start",)
    assert callback.events[-1] == ("end",)


def test_train_rejects_non_positive_epochs(gan_proxy, real_samples_loader, save_dir):
    trainer = GANTrainer(
        model_proxy=gan_proxy, train_data=real_samples_loader, save_dir_path=save_dir
    )

    with pytest.raises(ValueError):
        trainer.train(max_epochs=0)


def test_batch_as_tuple_is_unpacked_to_real_samples(gan_proxy, real_samples_loader, save_dir):
    trainer = GANTrainer(
        model_proxy=gan_proxy, train_data=real_samples_loader, save_dir_path=save_dir
    )

    result = trainer.train(max_epochs=1)

    assert result.epochs == 1


def test_result_model_save_path_exists(gan_proxy, real_samples_loader, save_dir):
    trainer = GANTrainer(
        model_proxy=gan_proxy, train_data=real_samples_loader, save_dir_path=save_dir
    )

    result = trainer.train(max_epochs=1)

    assert result.model_save_path.exists()


def test_unpack_real_batch_returns_bare_tensor_unchanged(gan_proxy, real_samples_loader, save_dir):
    trainer = GANTrainer(
        model_proxy=gan_proxy, train_data=real_samples_loader, save_dir_path=save_dir
    )
    batch = torch.randn(4, 4)

    result = trainer._unpack_real_batch(batch)

    assert result is batch


def test_show_progress_without_tqdm_does_not_raise(
    gan_proxy, real_samples_loader, save_dir, monkeypatch
):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("simulated missing tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    trainer = GANTrainer(
        model_proxy=gan_proxy,
        train_data=real_samples_loader,
        save_dir_path=save_dir,
        show_progress=True,
    )

    result = trainer.train(max_epochs=1)

    assert result.epochs == 1
