from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from src.logger import ModelConfig, ExperimentLogger

import numpy as np
import torch


class Train:
    def __init__(
        self,
        model: Module,
        optimiser: Optimizer,
        device: torch.device,
        config: ModelConfig,
        log: bool = True,
    ):
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = config.loss_fn
        self.device = device
        self.config = config
        self.epochs = config.epochs
        self.log = log

        self.logger = ExperimentLogger(config)

    def train(
        self, train: DataLoader, validation: DataLoader, test: DataLoader
    ) -> None:

        for epoch in range(self.epochs):
            avg_train_loss = self._train_one_epoch(train)
            avg_validation_loss = self._validate_one_epoch(validation)

            print(
                f"epoch: {epoch + 1} \t train loss: {avg_train_loss:.4f} \t"
                + f"validation loss: {avg_validation_loss:.4f}"
            )

        accuracy = self._test_one_epoch(test)
        print(f"Accuracy: {accuracy:.3f}%")

        if self.log:
            self.logger.save(accuracy)

    def _train_one_epoch(self, train: DataLoader) -> float:
        self.model.train()
        train_loss = []
        for data, label in train:
            data = data.to(device=self.device, dtype=torch.float32)
            label = label.to(device=self.device, dtype=torch.float32)

            loss = self.perform_one_step(data, label, training=True)
            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        self.logger.log_values([("train_loss", avg_loss)])
        return avg_loss

    def _validate_one_epoch(self, validation: DataLoader) -> float:
        self.model.eval()
        validation_loss = []
        with torch.no_grad():
            for data, label in validation:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)

                loss = self.perform_one_step(data, label, training=False)

                validation_loss.append(loss.item())

        avg_loss = np.mean(validation_loss)
        self.logger.log_values([("validation_loss", avg_loss)])
        return avg_loss

    def _test_one_epoch(self, test: DataLoader) -> float:
        total_tested = 0
        total_correct = 0

        self.model.eval()
        with torch.no_grad():
            for data, label in test:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)

                batch_tested, batch_correct = self.test_evaluation_logic(data, label)
                total_tested += batch_tested
                total_correct += batch_correct

        accuracy = (total_correct / total_tested) * 100

        return accuracy

    def perform_one_step(
        self, data: torch.tensor, label: torch.tensor, training: bool
    ) -> float:

        self.optimiser.zero_grad()
        pred = self.model(data).view(-1)
        labels = label.view(-1)

        loss = self.loss_fn(pred, labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()

        return loss

    def test_evaluation_logic(
        self, data: torch.tensor, label: torch.tensor
    ) -> tuple[int, int]:
        total_tested = 0
        total_correct = 0

        pred = self.model(data).view(-1)
        labels = label.view(-1)

        for pred, label in zip(pred, labels):
            pred = round(pred.item())
            label = round(label.item())
            print(f"pred: {pred} actual: {label}")
            total_tested += 1

            difference = abs(pred - label)

            if difference == 0:
                total_correct += 1

        return total_tested, total_correct
