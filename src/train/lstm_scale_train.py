from .train import Train

from typing import override
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from src.logger import ModelConfig, ExperimentLogger

import numpy as np
import torch


class LSTMScaleTrain(Train):
    def __init__(
        self,
        model: Module,
        optimiser: Optimizer,
        device: torch.device,
        config: ModelConfig,
        log: bool = True,
    ) -> None:
        super().__init__(model, optimiser, device, config, log)

    @override
    def _train_one_epoch(self, train: DataLoader) -> float:
        self.model.train()
        train_loss = []
        for data, label in train:
            data = data.to(device=self.device, dtype=torch.float32)
            label = label.to(device=self.device, dtype=torch.long)

            loss = self.perform_one_step(data, label, training=True)
            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        self.logger.log_values([("train_loss", avg_loss)])
        return avg_loss

    @override
    def _validate_one_epoch(self, validation: DataLoader) -> float:
        self.model.eval()
        validation_loss = []
        with torch.no_grad():
            for data, label in validation:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.long)

                loss = self.perform_one_step(data, label, training=False)

                validation_loss.append(loss.item())

        avg_loss = np.mean(validation_loss)
        self.logger.log_values([("validation_loss", avg_loss)])
        return avg_loss

    @override
    def perform_one_step(
        self, data: torch.tensor, label: torch.tensor, training: bool
    ) -> float:

        self.optimiser.zero_grad()
        pred = self.model(data)
        labels = label.view(-1)

        loss = self.loss_fn(pred, labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()

        return loss

    @override
    def test_evaluation_logic(
        self, data: torch.tensor, label: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:

        pred = self.model(data)
        labels = label.view(-1)

        _, predicted_classes = torch.max(pred, dim=1)

        for pred, label in zip(predicted_classes, labels):
            print(f"pred: {pred} actual: {label}")

        return predicted_classes, labels
