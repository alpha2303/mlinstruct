from datetime import datetime
from pathlib import Path

import torch
import numpy as np


class Trainer:
    """
    Creates a trainer that abstracts the model training loop.

    Required Arguments:
    - model - A PyTorch model object.
    - loss_fn - A PyTorch loss function object that returns an output that supports `loss.backward()` call.
    - optimizer - Optimizer object that inherits the `torch.optim.Optimizer` class.
    
    Optional Arguments:
    - scheduler - Learning rate scheduler object that inherits the `torch.optim.lr_scheduler.LRScheduler` class. Default: `None`
    - folder_path - A `pathlib.Path` object pointing to the file location for model checkpoint storage. Default: `None`
    - model_name - `str` that represents the name of the best model saved. Default: `None`
    """
    _TIMESTAMP_FORMAT = "%Y%m%d_%H%M"

    def __init__(self, model, loss_fn, optimizer, scheduler=None, folder_path=None, model_name=None):
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._early_stopper = None
        self._save_folder_path = None
        self._model_name = None
        self._start_epoch = 0
        self._training_timestamp = None
        
        self._initialize_checkpoints(folder_path=folder_path, model_name=model_name)
    
    def _train_one_epoch(self, trainLoader):
        running_loss = 0.
        last_loss = 0.
        self._model.train()
        
        for i, data in enumerate(trainLoader):
            X_batch, Y_batch = data
            Y_pred = self._model(X_batch)
            loss = self._loss_fn(Y_pred, Y_batch)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            
            running_loss += loss.item()

        return running_loss / len(trainLoader)
    
    def _validate(self, testLoader):
        running_vloss = 0.
        self._model.eval()
        
        with torch.no_grad():
            for i, vdata in enumerate(testLoader):
                vX_batch, vY_batch = vdata
                vY_pred = self._model(vX_batch)
                vloss = self._loss_fn(vY_pred, vY_batch)
                running_vloss += vloss.item()

        return running_vloss / len(testLoader)
    
    def _initialize_checkpoints(self, folder_path=None, model_name=None):
        if not folder_path:
            self._save_folder_path = Path(f"./Models/Train")
        else:
            self._save_folder_path = Path(folder_path)
        
        self._model_name = model_name if model_name else "best_model"
    
    def _save_model_checkpoint(self, epoch, loss):
        save_path = self._save_folder_path.joinpath(self._training_timestamp)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'loss': loss,
        }, save_path.joinpath(f'{self._model_name}.pt'))
    
    def get_save_folder_path(self):
        """
        Get `pathlib.Path` object of directory path where model checkpoints are saved.
        """
        if self._training_timestamp:
            return self._save_folder_path.joinpath(self._training_timestamp)
        else:
            return self._save_folder_path
    
    def allow_early_stop(self, patience=2., min_delta=0.):
        """
        Activate early stopping feature in training loop

        Arguments:
        - patience - `int` value representing the number of epochs to observe before stopping. Default: `2`
        - min_delta - `float` value that provides the minimum difference required between loss values required to consider early stopping. Default: `0.`

        """
        self._early_stopper = _EarlyStopper(patience=patience, min_delta=min_delta)
    
    def load_model_checkpoint(self, path_to_model):
        """
        Load a saved model checkpoint from file location provided.

        Arguments:
        - path_to_model - A `pathlib.Path` object of model checkpoint path.

        """
        checkpoint = torch.load(Path(path_to_model))
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._start_epoch = checkpoint['epoch']

        self._model.eval()

        return self._model
    
    def train(self, trainLoader, testLoader, n_iter):
        """
        Train the model stored in the Trainer object.

        Arguments:
        - trainLoader: A `torch.utils.data.DataLoader` object that contains the train Dataset
        - testLoader: A `torch.utils.data.DataLoader` object that contains the test Dataset
        - n_iter: Maximum number of training epochs

        """
        best_vloss = np.inf
        train_loss_list, test_loss_list = [], []
        self._training_timestamp = datetime.now().strftime(self._TIMESTAMP_FORMAT)
        
        for epoch_index in range(self._start_epoch, n_iter):
            avg_loss = self._train_one_epoch(trainLoader)
            avg_vloss = self._validate(testLoader)
        
            print(f"Epoch {epoch_index + 1}: Training Loss = {avg_loss} | Validation Loss = {avg_vloss} | Current LR = {self._optimizer.param_groups[0]['lr']}")
            train_loss_list.append(avg_loss)
            test_loss_list.append(avg_vloss)
            
            if self._scheduler:
                if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(avg_vloss)
                else:
                    self._scheduler.step()
            
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                self._save_model_checkpoint(epoch_index + 1, avg_vloss)
            
            if self._early_stopper and self._early_stopper.early_stop(avg_vloss):
                print(f"Early stopping triggered at epoch: {epoch_index + 1}")
                break
 
        return self._model, (train_loss_list, test_loss_list)

class _EarlyStopper:
    def __init__(self, patience = 2, min_delta = 0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_vloss = np.inf
    
    def early_stop(self, vloss):
        if vloss < self.min_vloss:
            self.min_vloss = vloss
            self.counter = 0
        elif vloss > (self.min_vloss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False