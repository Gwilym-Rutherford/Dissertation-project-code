from .train_base import Train

from typing import override
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from src.model.model_config_class import ModelConfig
from src.evaluation import Evaluation

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

    @override
    def test_and_evaluate_one_epoch(self, test: DataLoader) -> None:
        pred_list = []
        labels_list = []

        self.model.eval()
        with torch.no_grad():
            for data, label in test:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)

                pred_list.append(self.model(data))

                labels_list.append(label.view(-1))

        pred = torch.cat(pred_list).cpu()
        labels = torch.cat(labels_list).cpu()
    
        evaluation = Evaluation(pred, labels)

        if self.log:
            self.logger.save(evaluation, show_fig=self.verbose)
