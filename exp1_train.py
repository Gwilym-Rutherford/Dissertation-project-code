from DatasetManager import MSDataLoader, MileStone, Site
from torch.optim import Adam
from torch.nn import HuberLoss
from model.lstm_v1 import DMOLSTM

import numpy as np
import torch
import wandb

# wandb initialisation
wandb.init(project="LSTM on DMO data")

# setup constants
dmo_features = [
    "cadence_all_avg_d",
    "wbdur_all_avg_d",
    "strdur_30_avg_d",
    "cadence_30_avg_d",
    "ws_30_avg_d",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG_PATH = "config/config.yaml"
MILESTON = MileStone.T2
SITE = Site.MS10

INPUT_SIZE = len(dmo_features) * 2
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_DIM = 1

LEARNING_RATE = 1e-4

EPOCHS = 1000

# get dmo patient dataloaders
ms_dl = MSDataLoader(CONFIG_PATH)

training_dataloader, validation_dataloader, test_dataloader = (
    ms_dl.get_patient_dmo_dataloaders(MILESTON, SITE, dmo_features, BATCH_DIM)
)


print("Setup complete, training starting now")

# instantiate model, optimiser and loss function
model = DMOLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device=device)
optimiser = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = HuberLoss()


best_val_score = float("inf")
# train loop
for epoch in range(EPOCHS):
    model.train()
    for index, (input_data, label) in enumerate(training_dataloader):
        # ignore patients that have no available dmo data
        if input_data.shape == (1, 1):
            continue

        input_data = input_data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        output = model(input_data)
        loss = loss_fn(output, label)
        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for index, (input_data, label) in enumerate(validation_dataloader):
            if input_data.shape == (1, 1):
                continue

            input_data = input_data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output = model(input_data)
            loss = loss_fn(output, label)

            val_loss += loss.item()

    print(f"Epoch {epoch} | Train loss: {loss.item()} | Val loss: {val_loss}")

    # wandb log
    wandb.log({"loss": loss, "Accuracy": val_loss})

    if val_loss < best_val_score:
        best_val_score = val_loss
    else:
        print("stopping early")
        break


# final test
model.eval()
val_loss = []
with torch.no_grad():
    for index, (input_data, label) in enumerate(test_dataloader):
        if input_data.shape == (1, 1):
            continue

        input_data = input_data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        output = model(input_data)
        loss = loss_fn(output, label)

        val_loss.append(loss.item())

loss = np.array(val_loss)

print(f"final average loss: {np.mean(loss)}")

wandb.finish()
