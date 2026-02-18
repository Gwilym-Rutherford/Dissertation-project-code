from DatasetManager import MSDataLoader, MileStone, Site
from torch.optim import Adam
from torch.nn import HuberLoss
from model.lstm_v1 import DMOLSTM
from train_logic import DataLoaderConfig, Train

import numpy as np
import torch
import wandb

# wandb initialisation
# wandb.init(project="LSTM on DMO data")

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

train_config = DataLoaderConfig(
    training_dataloader, validation_dataloader, test_dataloader
)

# instantiate model, optimiser and loss function
model = DMOLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device=device)
optimiser = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = HuberLoss()

train_obj = Train(optimiser, loss_fn, model, device)

avg_loss = train_obj.train_one_epoch(train_config, lambda x: x.shape == (1, 1))
print(avg_loss)


# wandb.finish()
