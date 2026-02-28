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
        verbose: bool = True,
    ) -> None:
        super().__init__(model, optimiser, device, config, log, verbose)

        self.config.name = "lstm_regression_testing"
        self.logger = ExperimentLogger(self.config)

    @override
    def test_one_epoch(self, test: DataLoader) -> None:
        pred_list = []
        labels_list = []

        fatigue_scalar = 84

        self.model.eval()
        with torch.no_grad():
            for data, label in test:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)

                output = self.model(data).view(-1) * fatigue_scalar
                output = torch.round(output).long()

                labels = label.view(-1) * fatigue_scalar
                labels = torch.round(labels).long()

                pred_list.append(output)
                labels_list.append(labels)

        pred = torch.cat(pred_list).cpu()
        labels = torch.cat(labels_list).cpu()

        if self.log:
            self.logger.save(pred, labels, show_fig=self.verbose)

    # @override
    # def test_evaluation_logic(
    #     self, pred: torch.tensor, labels: torch.tensor
    # ) -> tuple[int, int]:
    #     threshold = 5
    #     fatigue_scalar = 84
    #     total_tested = 0
    #     total_correct = 0

    #     pred = pred * fatigue_scalar
    #     labels = labels * fatigue_scalar

    #     for pred, label in zip(pred, labels):
    #         pred = round(pred.item())
    #         label = round(label.item())
    #         print(f"pred: {pred} actual: {label}")
    #         total_tested += 1

    #         difference = abs(pred - label)

    #         if difference <= threshold:
    #             total_correct += 1

    #     return total_tested, total_correct
