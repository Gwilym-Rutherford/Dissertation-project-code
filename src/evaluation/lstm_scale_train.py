from .evalutation_base import Evaluation
from torchmetrics.classification import MulticlassF1Score
from sklearn import metrics

import matplotlib.pyplot as plt
import torch

class LSTMScaleEvaluation(Evaluation):
    def __init__(self, prediction: torch.Tensor, labels: torch.Tensor):
        super().__init__(prediction, labels)

    def compute_all_metrics(self) -> dict:

        metric_values = {}
        metric_values["accuracy"] = self.accuracy()
        metric_values["weighted F1"] = self.weighted_F1()
        metric_values["Macro F1"] = self.macro_F1()

        return metric_values


    def weighted_F1(self) -> float:
        recall = MulticlassF1Score(num_classes=self.NUM_CLASSES, average="weighted")
        return recall(self.preds, self.labels).item()

    def macro_F1(self) -> float:
        recall = MulticlassF1Score(num_classes=self.NUM_CLASSES, average="macro")
        return recall(self.preds, self.labels).item()

    def evaluation_plot(self):
        confusion_matrix = metrics.confusion_matrix(self.labels, self.preds)
        fig, ax = plt.subplots(figsize=(12, 12))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot(ax=ax, cmap='viridis', colorbar=True)
        plt.title("Model Confusion Matrix")
        ax.grid(False)
        return plt
    