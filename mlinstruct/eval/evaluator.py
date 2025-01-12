import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
)


class Evaluator:
    """
    Creates an evaluator that can generate metrics and plots.

    Arguments:
    - folder_path - A `pathlib.Path` object providing the directory path for storing generated plots.
    """

    def __init__(self, folder_path: Path):
        self.update_save_path(folder_path=folder_path)

    def update_save_path(self, folder_path: Path):
        """
        Update the file path where the generated plots are to be saved.

        Arguments:
        - folder_path - A `pathlib.Path` object providing the new directory path for storing generated plots.
        """
        self._folder_path = folder_path
        self._folder_path.mkdir(parents=True, exist_ok=True)

    def plot_loss(self, train_loss_list: list, validation_loss_list: list):
        """
        Generate and save plot of loss values generated during model training and validation to the save folder.

        The required input parameters are usually obtained from the output of
        `Trainer.train()` method and must have the same dimensions.

        Arguments:
        - train_loss_list - `list` of loss values per epoch generated during model training.
        - validation_loss_list - `list` of loss values per epoch generated during training validation.
        """
        plt.plot(train_loss_list, color="blue", label="train")
        plt.plot(validation_loss_list, color="orange", label="validation")
        plt.title("Train vs Validation Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper right")
        plt.savefig(self._folder_path.joinpath("loss.png"))

    def generate_roc_curve(self, Y_truth_tensor: torch.Tensor, y_pred: torch.Tensor):
        """
        Generate and save plot of ROC Curve obtained by comparing ground truth and predicted target values.
        This method also prints out the computed Area Under Curve (AUC) to `stdout`.

        The input parameters must have the same dimensions.

        Arguments:
        - Y_truth_tensor - A `torch.Tensor` object containing the ground truth target values
        - y_pred - A `torch.Tensor` object containing the target values predicted by the model.
        """
        fpr, tpr, thresholds = roc_curve(
            Y_truth_tensor.cpu().numpy(), y_pred.detach().cpu().numpy()
        )
        roc_auc = auc(fpr, tpr)

        print(f"AUC: {roc_auc}")

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (AUC = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic (ROC) curve")
        plt.legend(loc="lower right")
        plt.savefig(self._folder_path.joinpath("roc_curve.png"))

    def generate_confusion_matrix(
        self, Y_truth_tensor: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5
    ):
        """
        Generate and save plot of ROC Curve obtained by comparing ground truth and predicted target values.
        This method also prints out the computed accuracy and confusion matrix values to `stdout`.

        The input parameters must have the same dimensions.

        Arguments:
        - Y_truth_tensor - A `torch.Tensor` object containing the ground truth target values
        - y_pred - A `torch.Tensor` object containing the target values predicted by the model.
        """
        y_pred_final = [0 if y < threshold else 1 for y in y_pred]
        conf_mat = confusion_matrix(Y_truth_tensor.cpu().numpy(), y_pred_final)
        accuracy = accuracy_score(Y_truth_tensor.cpu().numpy(), y_pred_final)

        print(f"Accuracy:\n{accuracy}\n")
        print(f"Confusion Matrix:\n{conf_mat}\n")

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.annotate(
            text="Accuracy: %0.2f" % accuracy, xy=(40, 13), xycoords="figure pixels"
        )
        plt.savefig(self._folder_path.joinpath("confusion_matrix.png"))
