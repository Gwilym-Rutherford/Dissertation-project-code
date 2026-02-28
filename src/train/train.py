from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from src.logger import ModelConfig, ExperimentLogger
from contextlib import nullcontext

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
        verbose: bool = True,
    ):
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = config.loss_fn
        self.device = device
        self.config = config
        self.epochs = config.epochs
        self.log = log
        self.verbose = verbose

        self.logger = ExperimentLogger(config)

    def train(
        self, train: DataLoader, validation: DataLoader, test: DataLoader
    ) -> None:
        for epoch in range(self.epochs):
            avg_train_loss = self.run_one_epoch(train, training=True)
            avg_validation_loss = self.run_one_epoch(validation, training=False)

            if self.verbose:
                print(
                    f"epoch: {epoch + 1} \t train loss: {avg_train_loss:.4f} \t"
                    + f"validation loss: {avg_validation_loss:.4f}"
                )

        self.test_one_epoch(test)

    def run_one_epoch(self, dataset: DataLoader, training: bool) -> float:
        self.model.train() if training else self.model.eval()
        context = nullcontext() if training else torch.no_grad()
        loss = []

        with context:
            for data, label in dataset:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)

                loss_val = self.run_one_step(data, label, training=training)
                loss.append(loss_val.item())

        avg_loss = np.mean(loss)
        graph = "training_loss" if training else "validation_loss"
        self.logger.log_values([(graph, avg_loss)])
        return avg_loss

    def run_one_step(
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

    def test_one_epoch(self, test: DataLoader) -> None:
        pred_list = []
        labels_list = []

        self.model.eval()
        with torch.no_grad():
            for data, label in test:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)

                pred_list.append(self.model(data).view(-1))
                labels_list.append(label.view(-1))

        pred = torch.cat(pred_list).cpu()
        labels = torch.cat(labels_list).cpu()

        if self.log:
            self.logger.save(pred, labels, show_fig=self.verbose)


# class Train:
#     def __init__(
#         self,
#         model: Module,
#         optimiser: Optimizer,
#         device: torch.device,
#         config: ModelConfig,
#         log: bool = True,
#         verbose: bool = True
#     ):
#         self.model = model
#         self.optimiser = optimiser
#         self.loss_fn = config.loss_fn
#         self.device = device
#         self.config = config
#         self.epochs = config.epochs
#         self.log = log
#         self.verbose = verbose

#         self.logger = ExperimentLogger(config)

#     def train(
#         self, train: DataLoader, validation: DataLoader, test: DataLoader
#     ) -> None:

#         for epoch in range(self.epochs):
#             avg_train_loss = self._train_one_epoch(train)
#             avg_validation_loss = self._validate_one_epoch(validation)

#             if self.verbose:
#                 print(
#                     f"epoch: {epoch + 1} \t train loss: {avg_train_loss:.4f} \t"
#                     + f"validation loss: {avg_validation_loss:.4f}"
#                 )

#         self._test_one_epoch(test)

#     def _train_one_epoch(self, train: DataLoader) -> float:
#         self.model.train()
#         train_loss = []
#         for data, label in train:
#             data = data.to(device=self.device, dtype=torch.float32)
#             label = label.to(device=self.device, dtype=torch.float32)

#             loss = self.perform_one_step(data, label, training=True)
#             train_loss.append(loss.item())

#         avg_loss = np.mean(train_loss)
#         self.logger.log_values([("train_loss", avg_loss)])
#         return avg_loss

#     def _validate_one_epoch(self, validation: DataLoader) -> float:
#         self.model.eval()
#         validation_loss = []
#         with torch.no_grad():
#             for data, label in validation:
#                 data = data.to(device=self.device, dtype=torch.float32)
#                 label = label.to(device=self.device, dtype=torch.float32)

#                 loss = self.perform_one_step(data, label, training=False)

#                 validation_loss.append(loss.item())

#         avg_loss = np.mean(validation_loss)
#         self.logger.log_values([("validation_loss", avg_loss)])
#         return avg_loss

#     def _test_one_epoch(self, test: DataLoader) -> float:

#         pred_list = []
#         labels_list = []

#         self.model.eval()
#         with torch.no_grad():
#             for data, label in test:
#                 data = data.to(device=self.device, dtype=torch.float32)
#                 label = label.to(device=self.device, dtype=torch.float32)

#                 pred = self.model(data).view(-1)
#                 labels = label.view(-1)

#                 pred_list.append(pred)
#                 labels_list.append(labels)

#                 if self.verbose:
#                     self.print_pred_labels(pred, labels)

#         pred_tensor = torch.cat(pred_list, dim=0).cpu()
#         labels_tensor = torch.cat(labels_list, dim=0).cpu()

#         if self.log:
#             self.logger.save(pred_tensor, labels_tensor)


#     def perform_one_step(
#         self, data: torch.tensor, label: torch.tensor, training: bool
#     ) -> float:

#         self.optimiser.zero_grad()
#         pred = self.model(data).view(-1)
#         labels = label.view(-1)

#         loss = self.loss_fn(pred, labels)
#         if training:
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#             self.optimiser.step()

#         return loss

#     def print_pred_labels(
#         self, pred: torch.tensor, labels: torch.tensor
#     ) -> None:

#         for pred, label in zip(pred, labels):
#             pred = round(pred.item())
#             label = round(label.item())
#             print(f"pred: {pred} actual: {label}")
