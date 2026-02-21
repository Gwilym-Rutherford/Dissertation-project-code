from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from src.logger import ModelConfig, ExperimentLogger

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
    config: ModelConfig
):

    logger = ExperimentLogger(config)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        validation_loss = []
        for data, label in train:
            # if (data == 0).all() or torch.isnan(label).any():
            #     continue

            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            
            optimiser.zero_grad()

            pred = model(data).view(-1)
            labels = label.view(-1)
            
            loss = loss_fn(pred, labels)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            train_loss.append(loss.item())
        
        logger.log_values([("train_loss", np.average(train_loss))])

        model.eval()
        with torch.no_grad():
            for data, label in validation:
                # if (data == 0).all() or torch.isnan(label).any():
                #     continue


                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)


                pred = model(data).view(-1)
                labels = label.view(-1)
                loss = loss_fn(pred, labels)

                validation_loss.append(loss.item())

        logger.log_values([("validation_loss", np.average(validation_loss))])


        print(
           f"epoch: {epoch + 1} \t train loss: {np.average(train_loss):.4f} \t validation loss: {np.average(validation_loss):.4f}"
        )


    tolerance = 0.2
    total_tested = 0
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for data, label in test:
            if (data == 0).all() or torch.isnan(label).any():
                continue

            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred= model(data).view(-1)
            labels = label.view(-1)

            for pred, label in zip(pred, labels):
                print(f"pred: {pred} actual: {label}")
                total_tested += 1

            difference = abs(pred - label)

            if difference <= tolerance:
                total_correct += 1

    accuracy = (total_correct/total_tested) * 100

    print(f"Accuracy: {accuracy}%")
    logger.save(accuracy)