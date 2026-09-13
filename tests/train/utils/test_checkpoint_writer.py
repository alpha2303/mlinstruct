from pathlib import Path

from mlinstruct.train.utils.checkpoint_writer import CheckpointWriter


def test_checkpoint_filename_has_single_extension(save_dir, proxy):
    writer = CheckpointWriter(save_dir)
    writer.regenerate_model_save_path()
    checkpoint_path: Path = writer.create_checkpoint(proxy, epoch=1, vloss=0.5)

    assert checkpoint_path.suffix == ".pt"
    assert checkpoint_path.name.count(".pt") == 1


def test_create_checkpoint_returns_existing_path(save_dir, proxy):
    writer = CheckpointWriter(save_dir)
    writer.regenerate_model_save_path()
    checkpoint_path: Path = writer.create_checkpoint(proxy, epoch=1, vloss=0.5)

    assert checkpoint_path.exists()


def test_init_does_not_create_directory(save_dir):
    root = save_dir / "unused_root"
    CheckpointWriter(root)

    assert not root.exists()


def test_run_dir_uses_run_name(save_dir):
    writer = CheckpointWriter(save_dir, run_name="my-run")
    writer.regenerate_model_save_path()

    assert writer.get_model_save_path() == save_dir / "my-run"


def test_run_dir_collision_gets_suffix(save_dir):
    (save_dir / "my-run").mkdir(parents=True)

    writer = CheckpointWriter(save_dir, run_name="my-run")
    writer.regenerate_model_save_path()

    assert writer.get_model_save_path() == save_dir / "my-run_1"
