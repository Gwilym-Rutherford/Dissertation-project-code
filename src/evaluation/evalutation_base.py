from abc import ABC, abstractmethod
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np
import torch


class Evaluation(ABC):
    NUM_CLASSES = 85
    MAX_FATIGUE_SCORE = 84

    def __init__(self, predictions: torch.Tensor, labels: torch.Tensor):
        self.preds = predictions.squeeze() * Evaluation.MAX_FATIGUE_SCORE
        self.labels = labels * Evaluation.MAX_FATIGUE_SCORE

    def compute_all_metrics(self) -> dict:

        metric_values = {}
        metric_values["accuracy"] = self.accuracy()
        metric_values["Mean Squared Error"] = self.mean_squared_error()
        metric_values["Mean Absolute Error"] = self.mean_absolute_error()
        metric_values["R^2"] = self.R2()

        return metric_values

    def accuracy(self) -> float:
        threshold = 0.1 * Evaluation.MAX_FATIGUE_SCORE
        total_correct = 0

        for i in range(len(self.preds)):
            diff = abs(self.preds[i] - self.labels[i])
            if diff <= threshold:
                total_correct += 1

        return float(total_correct / len(self.preds))

    def mean_squared_error(self) -> float:
        mse = MeanSquaredError()
        return mse(self.preds, self.labels).item()

    def mean_absolute_error(self) -> float:
        mae = MeanAbsoluteError()
        return mae(self.preds, self.labels).item()

    def R2(self) -> float:
        r2 = R2Score()
        return r2(self.preds, self.labels).item()

    def evaluation_plot(self) -> plt:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(self.labels, self.preds, alpha=0.5, edgecolors="k")

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Perfect Prediction")

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs. Predicted (Regression)")
        ax.legend()
        return plt

    def bland_altman_plot(self) -> plt:
        fig, ax = plt.subplots(figsize=(8, 8))
        preds = self.preds
        labels = self.labels

        mean_val_x = (preds + labels) / 2
        diff_val_y = preds - labels
        ax.scatter(mean_val_x, diff_val_y)

        mean_diff = torch.mean(diff_val_y)
        std_diff = torch.std(diff_val_y)

        upper_limit = mean_diff + (1.96 * std_diff)
        lower_limit = mean_diff - (1.96 * std_diff)

        l_upper = ax.axhline(upper_limit, c="red", ls="dashed")
        l_lower = ax.axhline(lower_limit, c="red", ls="dashed")
        l_mean = ax.axhline(mean_diff, c="blue")
        l_center = ax.axhline(0, c="green")
        ax.legend(
            (l_upper, l_lower, l_mean, l_center),
            ("Upper limit", "Lower limit", "Mean", "Center"),
            loc="upper right",
        )
        ax.set_xlabel("Mean")
        ax.set_ylabel("Difference (prediction - label)")
        ax.set_title("Bland-Altman plot for prediction and ground truth")
        return plt

    def residual_plot(self) -> plt:
        fig, ax = plt.subplots(figsize=(8, 8))
        preds = self.preds
        labels = self.labels

        pred_val_x = labels
        diff_val_y = preds - labels
        ax.scatter(pred_val_x, diff_val_y)

        mean_diff = torch.mean(diff_val_y)

        l_mean = ax.axhline(mean_diff, c="blue")
        l_center = ax.axhline(0, c="green")
        ax.legend(
            (l_mean, l_center),
            ("Mean", "Center"),
            loc="upper right",
        )
        ax.set_xlabel("Predicted values")
        ax.set_ylabel("Difference (prediction - label)")
        ax.set_title("Residual Plot")
        return plt
