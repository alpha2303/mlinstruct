from typing import Iterable
import numpy as np

from ._base_trainer import BaseTrainer


class DefaultTrainer(BaseTrainer):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def train(self, max_epochs: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._model_proxy:
            raise Exception(
                "Model proxy is not provided. Chain the `add_model_proxy()` method to provide a model proxy."
            )

        if not self._train_data:
            raise Exception(
                "Training DataLoader is not provided. Chain the `add_train_data()` method to provide training DataLoader"
            )

        if not self._val_data:
            raise Exception(
                "Validation DataLoader is not provided. Chain the `add_val_data()` method to provide validation DataLoader"
            )

        best_vloss: float = np.inf
        train_loss_list, val_loss_list = [], []
        if self._checkpoint_writer:
            self._checkpoint_writer.regenerate_path()

        for epoch_index in range(0, max_epochs):
            avg_loss = self._model_proxy.train_one_epoch(self._train_data)

            avg_vloss = self._model_proxy.validate(self._val_data)

            print(
                f"Epoch {epoch_index + 1}: Training Loss = {avg_loss} | Validation Loss = {avg_vloss} | Learning Rate = {self._model_proxy.get_lr()}"
            )

            train_loss_list.append(avg_loss)
            val_loss_list.append(avg_vloss)

            if self._model_proxy.has_scheduler():
                self._model_proxy.step(avg_vloss=avg_loss)

            is_best: bool = False
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                is_best = True

            if self._checkpoint_writer:
                self._checkpoint_writer.create_checkpoint(
                    self._model_proxy, epoch_index + 1, avg_vloss, is_best
                )

            if self._early_stopper and self._early_stopper.early_stop(avg_vloss):
                print(f"Early stop triggered at epoch: {epoch_index + 1}")
                break
        
        if self._test_data:
            avg_tloss = self._model_proxy.validate(self._test_data)
            print(f"Average Test Loss: {avg_tloss}")

        if self._checkpoint_writer:
            print(
                f"Model checkpoints saved to {self._checkpoint_writer.get_model_save_path().resolve()}"
            )

        return (train_loss_list, val_loss_list)
