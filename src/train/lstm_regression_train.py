from .train import Train

from typing import override
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from src.logger import ModelConfig, ExperimentLogger

import numpy as np
import torch

class LSTMRegressionTrain(Train):
    def __init__(
        self,
        model: Module,
        optimiser: Optimizer,
        device: torch.device,
        config: ModelConfig,
        log: bool = True,
    ) -> None:
        super().__init__(model, optimiser, device, config, log)
        
        self.config.name = "lstm_regression_testing"
        self.logger = ExperimentLogger(self.config)
        

    @override
    def test_evaluation_logic(
        self, data: torch.tensor, label: torch.tensor
    ) -> tuple[int, int]:
        threshold = 5
        fatigue_scalar = 84
        total_tested = 0
        total_correct = 0

        pred = self.model(data).view(-1) * fatigue_scalar
        labels = label.view(-1) * fatigue_scalar

        for pred, label in zip(pred, labels):
            pred = round(pred.item())
            label = round(label.item())
            print(f"pred: {pred} actual: {label}")
            total_tested += 1

            difference = abs(pred - label)

            if difference <= threshold:
                total_correct += 1

        return total_tested, total_correct

