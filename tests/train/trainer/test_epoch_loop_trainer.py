import sys
import types

import pytest

from mlinstruct.train.callbacks import TrainerCallback
from mlinstruct.train.trainer.epoch_loop_trainer import EpochLoopTrainer
from mlinstruct.train.utils.checkpoint_writer import CheckpointWriter


class RecordingCallback(TrainerCallback):
    def __init__(self) -> None:
        self.events: list = []

    def on_train_start(self, trainer) -> None:
        self.events.append("start")

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss) -> None:
        self.events.append(f"epoch_end:{epoch}")

    def on_train_end(self, trainer, result) -> None:
        self.events.append("end")


class _RecordingEpochLoopTrainer(EpochLoopTrainer):
    def __init__(self, save_dir, stop_after=None, prepare_start_epoch=1, **kwargs):
        self.epochs_run: list[int] = []
        self.checkpoint_calls: list[tuple[int, bool]] = []
        self.after_epochs_call_count = 0
        self._stop_after = stop_after
        self._prepare_start_epoch = prepare_start_epoch
        super().__init__(checkpoint_writer=CheckpointWriter(save_dir), **kwargs)

    def _prepare_run(self) -> int:
        return self._prepare_start_epoch

    def _run_epoch(self, epoch_index):
        self.epochs_run.append(epoch_index)
        return float(epoch_index), float(-epoch_index)

    def _checkpoint_policy(self, epoch_index, metric_a, metric_b, is_final_epoch):
        self.checkpoint_calls.append((epoch_index, is_final_epoch))
        if is_final_epoch:
            path = self._checkpoint_writer.get_model_save_path() / f"epoch_{epoch_index}.ckpt"
            path.write_text("checkpoint")
            return path
        return None

    def _should_stop_early(self, metric_a, metric_b):
        return self._stop_after is not None and metric_a >= self._stop_after

    def _after_epochs(self):
        self.after_epochs_call_count += 1

    def _build_result(
        self,
        model_save_path,
        epochs_completed,
        metric_a_list,
        metric_b_list,
        metrics_history,
        best_checkpoint_path,
        stopped_early,
    ):
        return {
            "model_save_path": model_save_path,
            "epochs_completed": epochs_completed,
            "metric_a_list": metric_a_list,
            "metric_b_list": metric_b_list,
            "metrics_history": metrics_history,
            "best_checkpoint_path": best_checkpoint_path,
            "stopped_early": stopped_early,
        }


class _FailingEpochLoopTrainer(_RecordingEpochLoopTrainer):
    def _run_epoch(self, epoch_index):
        raise RuntimeError("boom")


def test_rejects_non_positive_epochs(save_dir):
    trainer = _RecordingEpochLoopTrainer(save_dir)

    with pytest.raises(ValueError):
        trainer.train(max_epochs=0)


def test_run_epoch_called_for_each_epoch(save_dir):
    trainer = _RecordingEpochLoopTrainer(save_dir)

    trainer.train(max_epochs=3)

    assert trainer.epochs_run == [1, 2, 3]


def test_checkpoint_policy_invoked_with_is_final_epoch_flag(save_dir):
    trainer = _RecordingEpochLoopTrainer(save_dir)

    trainer.train(max_epochs=3)

    assert trainer.checkpoint_calls == [(1, False), (2, False), (3, True)]


def test_should_stop_early_breaks_loop_early(save_dir):
    trainer = _RecordingEpochLoopTrainer(save_dir, stop_after=2)

    result = trainer.train(max_epochs=10)

    assert trainer.epochs_run == [1, 2]
    assert result["stopped_early"] is True


def test_callbacks_invoked_in_order(save_dir):
    callback = RecordingCallback()
    trainer = _RecordingEpochLoopTrainer(save_dir, callbacks=[callback])

    trainer.train(max_epochs=2)

    assert callback.events == ["start", "epoch_end:1", "epoch_end:2", "end"]


def test_metrics_history_collected_from_callbacks_with_history_attr(save_dir):
    class HistoryCallback(TrainerCallback):
        def __init__(self):
            self.history = {"metric": []}

        def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
            self.history["metric"].append(train_loss)

    trainer = _RecordingEpochLoopTrainer(save_dir, callbacks=[HistoryCallback()])

    result = trainer.train(max_epochs=2)

    assert result["metrics_history"] == {"metric": [1.0, 2.0]}


def test_after_epochs_hook_called_once(save_dir):
    trainer = _RecordingEpochLoopTrainer(save_dir)

    trainer.train(max_epochs=3)

    assert trainer.after_epochs_call_count == 1


def test_prepare_run_controls_start_epoch(save_dir):
    trainer = _RecordingEpochLoopTrainer(save_dir, prepare_start_epoch=3)

    result = trainer.train(max_epochs=5)

    assert trainer.epochs_run == [3, 4, 5]
    assert result["epochs_completed"] == 5


def test_error_during_run_epoch_is_logged_and_reraised(save_dir):
    trainer = _FailingEpochLoopTrainer(save_dir)

    with pytest.raises(RuntimeError, match="boom"):
        trainer.train(max_epochs=1)


def test_show_progress_with_tqdm_installed_wraps_epoch_iterator(save_dir, monkeypatch):
    tqdm_calls: list[dict] = []

    def fake_tqdm(iterable, desc=None):
        tqdm_calls.append({"iterable": iterable, "desc": desc})
        return iterable

    fake_tqdm_module = types.SimpleNamespace(tqdm=fake_tqdm)
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm_module)

    trainer = _RecordingEpochLoopTrainer(save_dir, show_progress=True)

    trainer.train(max_epochs=2)

    assert trainer.epochs_run == [1, 2]
    assert len(tqdm_calls) == 1
    assert tqdm_calls[0]["desc"] == "Training"
    assert list(tqdm_calls[0]["iterable"]) == [1, 2]
