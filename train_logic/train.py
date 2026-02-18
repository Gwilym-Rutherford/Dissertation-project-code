from .train_config import DataLoaderConfig

import numpy as np
import torch
import wandb


class Train:
    def __init__(
        self,
        optimiser: torch.optim.Optimizer,
        loss_fn: callable,
        model: torch.nn.Module,
        device: torch.device,
        lr: float = 1e-4,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.device = device

    def train_one_epoch(
        self, dataloaders: DataLoaderConfig, input_filter: callable
    ) -> float:

        training_dataloader = dataloaders.train_dataloader
        validation_dataloader = dataloaders.validation_dataloader
        test_dataloader = dataloaders.test_dataloader

        avg_loss = []

        self.model.train()
        for input_data, label in training_dataloader:
            if input_filter is not None and input_filter(input_data):
                continue

            input_data = input_data.to(device=self.device, dtype=torch.float32)
            label = label.to(device=self.device, dtype=torch.float32)

            output = self.model(input_data)
            loss = self.loss_fn(output, label)
            loss.backward()

            self.optimiser.step()
            self.optimiser.zero_grad()

            avg_loss.append(loss)

        return sum(avg_loss) / len(avg_loss)


# best_val_score = float("inf")
# # train loop
# for epoch in range(EPOCHS):
#     model.train()
#     for index, (input_data, label) in enumerate(training_dataloader):
#         # ignore patients that have no available dmo data
#         if input_data.shape == (1, 1):
#             continue

#         input_data = input_data.to(device=device, dtype=torch.float32)
#         label = label.to(device=device, dtype=torch.float32)

#         output = model(input_data)
#         loss = loss_fn(output, label)
#         loss.backward()

#         optimiser.step()
#         optimiser.zero_grad()

#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for index, (input_data, label) in enumerate(validation_dataloader):
#             if input_data.shape == (1, 1):
#                 continue

#             input_data = input_data.to(device=device, dtype=torch.float32)
#             label = label.to(device=device, dtype=torch.float32)

#             output = model(input_data)
#             loss = loss_fn(output, label)

#             val_loss += loss.item()

#     print(f"Epoch {epoch} | Train loss: {loss.item()} | Val loss: {val_loss}")

#     # wandb log
#     wandb.log({"loss": loss, "Accuracy": val_loss})

#     if val_loss < best_val_score:
#         best_val_score = val_loss
#     else:
#         print("stopping early")
#         break


# # final test
# model.eval()
# val_loss = []
# with torch.no_grad():
#     for index, (input_data, label) in enumerate(test_dataloader):
#         if input_data.shape == (1, 1):
#             continue

#         input_data = input_data.to(device=device, dtype=torch.float32)
#         label = label.to(device=device, dtype=torch.float32)

#         output = model(input_data)
#         loss = loss_fn(output, label)

#         val_loss.append(loss.item())

# loss = np.array(val_loss)

# print(f"final average loss: {np.mean(loss)}")

# wandb.finish()
