from .evalutation_base import Evaluation
from sklearn import metrics

import matplotlib.pyplot as plt
import torch

class LSTMScaleEvaluation(Evaluation):
    def __init__(self, prediction: torch.Tensor, labels: torch.Tensor):
        super().__init__(prediction, labels)

    def evaluation_plot(self):
        confusion_matrix = metrics.confusion_matrix(self.labels, self.pred)
        fig, ax = plt.subplots(figsize=(12, 12))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot(ax=ax, cmap='viridis', colorbar=True)
        plt.title("Model Confusion Matrix")
        ax.grid(False)
        return plt
    