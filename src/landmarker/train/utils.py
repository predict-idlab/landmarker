"""Utils for training."""

import os
import random
from typing import Optional

import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping to stop the training when the score does not improve after
    certain epochs.
    source: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

    Args:
        patience (int, optional): Number of epochs to wait for the score to
            improve. Defaults to 25.
        min_delta (float, optional): Minimum difference between new score and
            old score for new score to be considered as an improvement.
            Defaults to 0.0.
        verbose (bool, optional): Whether to print the logs or not.
            Defaults to False.
        greater_is_better (bool, optional): Whether the new score is expected
            to be greater than previous scores or not.
            Defaults to False.
        name_score (str, optional): Name of the score being tracked.
            Defaults to 'Val Loss'.
    """

    def __init__(
        self,
        patience: int = 25,
        min_delta: float = 0.0,
        verbose: bool = False,
        greater_is_better: bool = False,
        name_score: str = "Val Loss",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.greater_is_better = greater_is_better
        self.name_score = name_score

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_score: float) -> None:
        """
        Args:
            val_score (float): Validation score to compare for early stopping.
        """
        if self.best_score is None:
            self.best_score = val_score
        elif self.best_score - val_score > self.min_delta and (not self.greater_is_better):
            self.best_score = val_score
            # reset counter if validation loss improves
            self.counter = 0
        elif val_score - self.best_score > self.min_delta and (self.greater_is_better):
            self.best_score = val_score
            # reset counter if validation loss improves
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"INFO: Early stopping counter \
                    {self.counter} of {self.patience} \
                        [Best {self.name_score}: {self.best_score:.8f}, \
                            {self.name_score}: {val_score:.8f}]"
                )
            if self.counter >= self.patience:
                if self.verbose:
                    print("INFO: Early stopping DONE!")
                    print(f"INFO: Best {self.name_score}: {self.best_score:.8f}")
                self.early_stop = True


class SaveBestModel:
    """
    Save the best model based on validation metric/loss.

    Args:
        verbose (bool, optional): Whether to print the logs or not.
            Defaults to False.
        greater_is_better (bool, optional): Whether the new score is expected
            to be greater than previous scores or not.
            Defaults to False.
        name_score (str, optional): Name of the score being tracked.
            Defaults to 'Val Loss'.
    """

    def __init__(
        self, verbose: bool = False, greater_is_better: bool = False, name_score: str = "Val_Loss"
    ) -> None:
        self.verbose = verbose
        self.best_score: Optional[float] = None
        self.model_id = random.randint(0, 100000)
        while os.path.exists(f"best_weights_{self.model_id}_{name_score.lower()}.pt"):
            self.model_id = random.randint(0, 100000)
        self.greater_is_better = greater_is_better
        self.name_score = name_score
        self.path = f"best_weights_{self.model_id}_{self.name_score.lower()}.pt"

    def __call__(self, val_score: float, model: torch.nn.Module) -> None:
        """
        Args:
            val_score (float): Validation score to compare for saving the model.
            model (torch.nn.Module): Model to save.
        """
        if self.best_score is None:
            self.best_score = val_score
            # Save model
            torch.save(model.state_dict(), self.path)
        elif (val_score < self.best_score) and (not self.greater_is_better):
            self.save_checkpoint(val_score, model)
        elif (val_score > self.best_score) and (self.greater_is_better):
            self.save_checkpoint(val_score, model)

    def save_checkpoint(self, val_score: float, model: torch.nn.Module) -> None:
        """
        Args:
            val_score (float): Validation score to compare for saving the model.
            model (torch.nn.Module): Model to save.
        """
        if self.verbose:
            print(
                f"INFO: {self.name_score} improved ({self.best_score:.8f} --> {val_score:.8f}). "
                + "Saving model ..."
            )
        self.best_score = val_score
        # Save model
        torch.save(model.state_dict(), self.path)


def set_seed(seed: int = 1817) -> None:
    """
    Set seed for reproducibility.
        source:
        https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
