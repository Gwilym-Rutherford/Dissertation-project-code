from torchmetrics.classification import MulticlassRecall, MulticlassF1Score
from sklearn import metrics

import matplotlib.pyplot as plt
import torch

NUM_CLASSES = 85

def compute_all_metrics(pred: torch.tensor, labels: torch.tensor) -> None:

    metrics = {}
    metrics["accuracy"] = accuracy(pred, labels)
    metrics["recall"] = recall(pred, labels)
    metrics["weighted F1"] = weighted_F1(pred, labels)
    metrics["Macro F1"] = macro_F1(pred, labels)

    return metrics

def accuracy(pred: torch.tensor, labels: torch.tensor) -> str:
    total_correct = (pred == labels).sum().item()
    total_tested = labels.size(0)
    return f"{(total_correct / total_tested):.3f}%"


def recall(pred: torch.tensor, label: torch.tensor) -> float:
    recall = MulticlassRecall(num_classes=NUM_CLASSES, average="macro")
    return recall(pred, label).item()


def weighted_F1(pred: torch.tensor, label: torch.tensor) -> float:
    recall = MulticlassF1Score(num_classes=NUM_CLASSES, average="weighted")
    return recall(pred, label).item()


def macro_F1(pred, label) -> float:
    recall = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
    return recall(pred, label).item()


def confusion_matrix(pred, label) -> plt:
    confusion_matrix = metrics.confusion_matrix(label, pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot(ax=ax, cmap='viridis', colorbar=True)
    plt.title("Model Confusion Matrix")
    ax.grid(False)
    # cm_display.plot()
    return plt


# maybe
def balanced_accuracy(pred, label):
    pass
