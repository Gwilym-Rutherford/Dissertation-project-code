from src.patient_data_dispatcher import PatientDataDispatcher, PatientDataType
from src.core.enums import MileStone, UniformMethod
from src.pipeline import dmo_into_dataloader
from src.pipeline.dmo_train_catagory import dmo_train_catagory
from src.logger import ModelConfig
from src.model import DMOLSTM
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from src.research_util import plot_distribution

import time
import torch

# for debuggin print matrix nicely with no line breaks
torch.set_printoptions(linewidth=200, sci_mode=False, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get respected data
dmo_features = [
    "wb_all_sum_d",
    "walkdur_all_sum_d",
    "wbsteps_all_sum_d",
    "wbdur_all_avg_d",
    "wbdur_all_p90_d",
    "wbdur_all_var_d",
    "cadence_all_avg_d",
    "strdur_all_avg_d",
    "cadence_all_var_d",
    "strdur_all_var_d",
    "ws_1030_avg_d",
    "strlen_1030_avg_d",
    "wb_10_sum_d",
    "ws_10_p90_d",
    "wb_30_sum_d",
    "ws_30_avg_d",
    "strlen_30_avg_d",
    "cadence_30_avg_d",
    "strdur_30_avg_d",
    "ws_30_p90_d",
    "cadence_30_p90_d",
    "ws_30_var_d",
    "strlen_30_var_d",
    "wb_60_sum_d",
    "total_worn_h_d",
    "total_worn_during_waking_h_d",
]

# get dmo data and labels
print("getting data")
pdd = PatientDataDispatcher("config/config.yaml", dmo_features, MileStone.T1)
ids = list(set(pdd.metadata["Local.Participant"].to_list()))
dmo_data, dmo_labels = pdd.get_patient_data(PatientDataType.DMO, ids=ids)

# setup training params
input_size = len(dmo_features)
hidden_size = 128
num_layers = 2
output_size = 10

lr = 5e-4
loss_fn = CrossEntropyLoss()
# loss_fn = MSELoss()

epochs = 200
batch_size = 1

# load into dataloaders
print("loading into dataloaders")
train, validation, test = dmo_into_dataloader(
    dmo_data, dmo_labels, batch_size=batch_size, uniform_method=UniformMethod.UPSAMPLE
)

model = DMOLSTM(input_size, hidden_size, num_layers, output_size).to(device=device)
optimiser = Adam(model.parameters(), lr=lr)


config = ModelConfig(
    name="lstm_training",
    model_type="LSTM",
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    epochs=epochs,
    optimiser=str(optimiser),
    loss_fn=str(loss_fn),
    learning_rate=lr,
    notes=f"loss_fn: {str(loss_fn)}    lr: {lr}    train samples: {len(train) * batch_size}    epochs: {epochs}    batch size: {batch_size}",
)
print("Beginning training")
dmo_train_catagory(model, optimiser, loss_fn, epochs, device, train, validation, test, config)
