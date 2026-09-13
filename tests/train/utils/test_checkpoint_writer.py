from pathlib import Path

from mlinstruct.train.utils.checkpoint_writer import CheckpointWriter


def test_checkpoint_filename_has_single_extension(save_dir, proxy):
    writer = CheckpointWriter(save_dir)
    checkpoint_path: Path = writer.create_checkpoint(proxy, epoch=1, vloss=0.5)

    assert checkpoint_path.suffix == ".pt"
    assert checkpoint_path.name.count(".pt") == 1


def test_create_checkpoint_returns_existing_path(save_dir, proxy):
    writer = CheckpointWriter(save_dir)
    checkpoint_path: Path = writer.create_checkpoint(proxy, epoch=1, vloss=0.5)

    assert checkpoint_path.exists()
