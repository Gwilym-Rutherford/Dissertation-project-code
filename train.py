from src.patient_data_dispatcher import PatientDataDispatcher, PatientDataType
from src.core.enums import MileStone
from src.pipeline import dmo_into_dataloader
from src.pipeline.dmo_train import dmo_train
from src.logger import ModelConfig
from src.model import DMOLSTM
from torch.optim import Adam
from torch.nn import HuberLoss

import torch

# for debuggin print matrix nicely with no line breaks
torch.set_printoptions(linewidth=200, sci_mode=False, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get respected data
dmo_features = [
    "cadence_all_avg_d",
    "wbdur_all_avg_d",
    "strdur_30_avg_d",
    "cadence_30_avg_d",
    "ws_30_avg_d",
]

pdd = PatientDataDispatcher("config/config.yaml", dmo_features, MileStone.T2)

metadata = pdd.get_patient_data(PatientDataType.META)

# print(metadata)
all_ids = list(set(metadata["Local.Participant"].to_list()))

patient_label = pdd.get_patient_data(PatientDataType.META, ids=all_ids[:100])

fatigue_df = patient_label[["Local.Participant", "visit.number", "MFISTO1N"]]

dmo_labels = torch.tensor(fatigue_df["MFISTO1N"].tolist())
dmo_data = pdd.get_patient_data(PatientDataType.DMO, ids=all_ids[:100])

train, validation, test = dmo_into_dataloader(dmo_data, dmo_labels, batch_size=1)

input_size = len(dmo_features)
hidden_size = 128
num_layers = 1
output_size = 1

lr = 1e-4
loss_fn = HuberLoss()

epochs = 200

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
    notes="No notes",
)

dmo_train(model, optimiser, loss_fn, epochs, device, train, validation, test, config)
