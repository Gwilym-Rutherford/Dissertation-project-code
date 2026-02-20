from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from src.logger.grapher import Grapher

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

    logger = Grapher("lstm_training", "Baseline results")


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
        
        logger.log_values([("train_loss", np.average(train_loss))])

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

        logger.log_values([("validation_loss", np.average(validation_loss))])


        print(
            f"epoch: {epoch} \t train loss: {np.average(train_loss):.4f} \t validation loss: {np.average(validation_loss):.4f}"
        )


    tolerance = 0.2
    total_tested = 0
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for data, label in test:
            if (data == 0).all():
                continue

            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            
            y_hat= model(data).item()
            y = label.item()

            print(f"pred: {y_hat} actual: {y}")

            difference = abs(y_hat - y)

            total_tested += 1    
            if difference <= tolerance:
                total_correct += 1

    accuracy = (total_correct/total_tested) * 100

    print(f"Accuracy: {accuracy}%")
    logger.make_graph()