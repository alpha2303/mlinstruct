import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from mlinstruct.train.callbacks import TrainerCallback
from mlinstruct.train.gan_train_result import GANTrainResult
from mlinstruct.train.model_proxy.gan_model_proxy import GANModelProxy
from mlinstruct.train.trainer.base_trainer import DEFAULT_SAVE_PATH
from mlinstruct.train.trainer.epoch_loop_trainer import EpochLoopTrainer
from mlinstruct.train.utils.checkpoint_writer import CheckpointWriter


class GANTrainer(EpochLoopTrainer):
    """Trains a vanilla (unconditional, single-G/single-D) GAN via a GANModelProxy.

    Does not subclass BaseTrainer: a GAN has no single scalar validation loss to
    checkpoint or early-stop on, so checkpointing here is cadence-based
    (checkpoint_interval, plus always the final epoch) rather than best-loss-based,
    and no EarlyStopper is offered. Shares its epoch-loop/callback/checkpoint-writer
    scaffolding with DefaultTrainer via EpochLoopTrainer.

    Reuses TrainerCallback.on_epoch_end(trainer, epoch, train_loss, val_loss) as-is,
    passing train_loss=avg_generator_loss, val_loss=avg_discriminator_loss, so every
    existing TrainerCallback subclass (and the hasattr(callback, "history")
    metrics_history duck-typing) works against GANTrainer unmodified.

    Args:
        model_proxy (GANModelProxy): Proxy of the generator/discriminator pair to train.
        train_data (Iterable): Yields batches of real samples: a bare Tensor, or an
            (x, ...) tuple/list (what an unlabeled or labeled DataLoader both
            naturally produce) — batch[0] is used when the batch is a tuple/list,
            else the batch itself.
        n_critic (int): Number of discriminator steps to run per generator step.
        checkpoint_interval (int): Checkpoint every this many epochs. The final
            epoch is always checkpointed regardless of this interval.
        save_dir_path (Path): Root directory path for saving model checkpoints.
        run_name (Optional[str]): Name for this run's checkpoint directory.
            Defaults to a timestamp; a name that already exists under
            save_dir_path gets a numeric suffix.
        logger (Optional[logging.Logger]): Logger for logging training progress.
            Defaults to the module logger.
        callbacks (Sequence[TrainerCallback]): Callbacks invoked at the start of
            training, at the end of every epoch, and at the end of training.
        show_progress (bool): Whether to show a tqdm progress bar over epochs.
            Silently does nothing if tqdm isn't installed. Defaults to False.
    """

    def __init__(
        self,
        model_proxy: GANModelProxy,
        train_data: Iterable,
        n_critic: int = 1,
        checkpoint_interval: int = 1,
        save_dir_path: Path = DEFAULT_SAVE_PATH,
        run_name: str | None = None,
        logger: logging.Logger | None = None,
        callbacks: Sequence[TrainerCallback] = (),
        show_progress: bool = False,
    ) -> None:
        self._model_proxy = model_proxy
        self._train_data = train_data
        self._n_critic = n_critic
        self._checkpoint_interval = checkpoint_interval
        super().__init__(
            checkpoint_writer=CheckpointWriter(save_dir_path, run_name=run_name),
            callbacks=callbacks,
            show_progress=show_progress,
            logger=logger,
        )

    def _unpack_real_batch(self, batch):
        if isinstance(batch, tuple | list):
            return batch[0]
        return batch

    def train(self, max_epochs: int) -> GANTrainResult:
        """Train the GAN.

        Args:
            max_epochs (int): The number of epochs to train for.

        Returns:
            GANTrainResult: The result of the training process.
        """
        return super().train(max_epochs)

    def _run_epoch(self, epoch_index: int) -> tuple[float, float]:
        running_g_loss = 0.0
        running_d_loss = 0.0
        num_batches = 0

        for batch in self._train_data:
            real_batch = self._unpack_real_batch(batch)
            g_loss, d_loss = self._model_proxy.train_one_batch(real_batch, n_critic=self._n_critic)
            running_g_loss += g_loss
            running_d_loss += d_loss
            num_batches += 1

        avg_g_loss = running_g_loss / num_batches
        avg_d_loss = running_d_loss / num_batches

        self._logger.info(
            f"Epoch {epoch_index}: Generator Loss = {avg_g_loss} | "
            f"Discriminator Loss = {avg_d_loss}"
        )

        return avg_g_loss, avg_d_loss

    def _checkpoint_policy(
        self, epoch_index: int, metric_a: float, metric_b: float, is_final_epoch: bool
    ) -> Path | None:
        avg_g_loss, avg_d_loss = metric_a, metric_b
        if epoch_index % self._checkpoint_interval == 0 or is_final_epoch:
            stem = f"model_epoch_{epoch_index}_gloss_{avg_g_loss:.4f}_dloss_{avg_d_loss:.4f}"
            return self._model_proxy.save_weights(
                epoch_index,
                self._checkpoint_writer.get_model_save_path(),  # type: ignore
                stem,
                g_loss=avg_g_loss,
                d_loss=avg_d_loss,
            )
        return None

    def _build_result(
        self,
        model_save_path: Path,
        epochs_completed: int,
        metric_a_list: list[float],
        metric_b_list: list[float],
        metrics_history: dict[str, list[float]],
        best_checkpoint_path: Path | None,
        stopped_early: bool,
    ) -> GANTrainResult:
        return GANTrainResult(
            model_name=self._model_proxy.get_model_name(),
            model_save_path=model_save_path,
            epochs=epochs_completed,
            g_loss_list=metric_a_list,
            d_loss_list=metric_b_list,
            metrics_history=metrics_history,
        )
