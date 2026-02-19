from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

import numpy as np
import torch


def dmo_train(
    model: Module,
    optimiser: Optimizer,
    loss_fn: callable,
    epochs: int,
    device: torch.device,
    train: DataLoader,
    validation: DataLoader,
    test: DataLoader,
):

    val_loss_avg = []

    for epoch in range(epochs):
        train_loss = []
        validation_loss = []

        model.train()
        for data, label in train:
            if (data == 0).all():
                continue

            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()

            optimiser.step()
            optimiser.zero_grad()

            train_loss.append(loss.cpu().item())

        model.eval()
        with torch.no_grad():
            for data, label in validation:
                if (data == 0).all():
                    continue

                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                output = model(data)
                loss = loss_fn(output, label)

                validation_loss.append(loss.cpu().item())

        print(
            f"epoch: {epoch} \t train loss: {np.average(train_loss):.4f} \t validation loss: {np.average(validation_loss):.4f}"
        )
