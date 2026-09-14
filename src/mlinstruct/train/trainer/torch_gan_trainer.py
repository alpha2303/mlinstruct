import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from mlinstruct.train.callbacks import TrainerCallback
from mlinstruct.train.gan_train_result import GANTrainResult
from mlinstruct.train.model_proxy.torch_gan_model_proxy import GANModelProxy
from mlinstruct.train.trainer.base_trainer import DEFAULT_SAVE_PATH
from mlinstruct.train.utils.checkpoint_writer import CheckpointWriter


class GANTrainer:
    """Trains a vanilla (unconditional, single-G/single-D) GAN via a GANModelProxy.

    Does not subclass BaseTrainer: a GAN has no single scalar validation loss to
    checkpoint or early-stop on, so checkpointing here is cadence-based
    (checkpoint_interval, plus always the final epoch) rather than best-loss-based,
    and no EarlyStopper is offered.

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
        self._checkpoint_writer = CheckpointWriter(save_dir_path, run_name=run_name)
        self._logger = logger or logging.getLogger(__name__)
        self._callbacks = callbacks
        self._show_progress = show_progress

    def _epoch_iterator(self, epochs: range):
        if not self._show_progress:
            return epochs

        try:
            from tqdm import tqdm
        except ImportError:
            return epochs

        return tqdm(epochs, desc="Training")

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
        if max_epochs <= 0:
            raise ValueError("Max epochs must be positive number greater than 0.")

        self._checkpoint_writer.regenerate_model_save_path()  # type: ignore

        g_loss_list: list[float] = []
        d_loss_list: list[float] = []

        for callback in self._callbacks:
            callback.on_train_start(self)

        for epoch_index in self._epoch_iterator(range(1, max_epochs + 1)):
            running_g_loss = 0.0
            running_d_loss = 0.0
            num_batches = 0

            for batch in self._train_data:
                real_batch = self._unpack_real_batch(batch)
                g_loss, d_loss = self._model_proxy.train_one_batch(
                    real_batch, n_critic=self._n_critic
                )
                running_g_loss += g_loss
                running_d_loss += d_loss
                num_batches += 1

            avg_g_loss = running_g_loss / num_batches
            avg_d_loss = running_d_loss / num_batches

            self._logger.info(
                f"Epoch {epoch_index}: Generator Loss = {avg_g_loss} | "
                f"Discriminator Loss = {avg_d_loss}"
            )

            g_loss_list.append(avg_g_loss)
            d_loss_list.append(avg_d_loss)

            for callback in self._callbacks:
                callback.on_epoch_end(self, epoch_index, avg_g_loss, avg_d_loss)

            if epoch_index % self._checkpoint_interval == 0 or epoch_index == max_epochs:
                stem = f"model_epoch_{epoch_index}_gloss_{avg_g_loss:.4f}_dloss_{avg_d_loss:.4f}"
                self._model_proxy.save_weights(
                    epoch_index,
                    self._checkpoint_writer.get_model_save_path(),  # type: ignore
                    stem,
                    g_loss=avg_g_loss,
                    d_loss=avg_d_loss,
                )

        model_save_path = self._checkpoint_writer.get_model_save_path().resolve()  # type: ignore
        self._logger.info(f"Model checkpoints saved to {model_save_path}")

        metrics_history: dict[str, list[float]] = {}
        for callback in self._callbacks:
            if hasattr(callback, "history"):
                metrics_history.update(callback.history)

        result = GANTrainResult(
            model_name=self._model_proxy.get_model_name(),
            model_save_path=model_save_path,
            epochs=len(g_loss_list),
            g_loss_list=g_loss_list,
            d_loss_list=d_loss_list,
            metrics_history=metrics_history,
        )

        for callback in self._callbacks:
            callback.on_train_end(self, result)

        return result
