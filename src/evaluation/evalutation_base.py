from abc import ABC, abstractmethod
from torchmetrics.classification import MulticlassRecall, MulticlassF1Score
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np
import torch

class Evaluation(ABC):
    NUM_CLASSES = 85

    def __init__(self, prediction: torch.Tensor, labels: torch.Tensor):
        self.pred = prediction.squeeze()
        self.labels = labels

    def compute_all_metrics(self) -> dict:

        metric_values = {}
        metric_values["accuracy"] = self.accuracy()
        metric_values["Mean Squared Error"] = self.mean_squared_error()
        metric_values["Mean Absolute Error"] = self.weighted_F1()
        metric_values["R^2"] = self.weighted_F1()
        metric_values["weighted F1"] = self.weighted_F1()
        metric_values["Macro F1"] = self.macro_F1()

        return metric_values

    def accuracy(self) -> float:
        # total_correct = (self.pred == self.labels).sum().item()
        # total_tested = self.labels.size(0)
        # return f"{(total_correct / total_tested):.3f}%"
        threshold = 0.1
        total_correct = 0

        for i in range(len(self.pred)):
            diff = abs(self.pred[i] - self.labels[i])
            if diff <= threshold:
                total_correct += 1

        return float(total_correct/len(self.pred))

    def mean_squared_error(self) -> float:
        mse = MeanSquaredError()
        return mse(self.pred, self.labels).item()

    def mean_absolute_error(self) -> float:
        mae = MeanAbsoluteError()
        return mae(self.pred, self.labels).item()

    def R2(self) -> float:
        r2 = R2Score()
        return r2(self.pred, self.labels).item()

    def weighted_F1(self) -> float:
        recall = MulticlassF1Score(num_classes=self.NUM_CLASSES, average="weighted")
        return recall(self.pred, self.labels).item()

    def macro_F1(self) -> float:
        recall = MulticlassF1Score(num_classes=self.NUM_CLASSES, average="macro")
        return recall(self.pred, self.labels).item()

    def evaluation_plot(self) -> plt:
        fig, ax = plt.subplots(figsize=(8, 8))
    
        ax.scatter(self.labels, self.pred, alpha=0.5, edgecolors='k')
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),  
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Prediction")
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs. Predicted (Regression)')
        ax.legend()
        return plt
