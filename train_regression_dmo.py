from src.patient_data_dispatcher import PatientDataDispatcher, PatientDataType
from src.core.enums import MileStone, UniformMethod
from src.pipeline import dmo_into_dataloader
from src.model import DMOLSTM
from src.model import lstm_regression
from src.train import LSTMRegressionTrain
from torchvision.transforms import Compose
from src.core.data_transforms import Transform

import torch

# for debuggin print matrix nicely with no line breaks
torch.set_printoptions(linewidth=200, sci_mode=False, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

print("getting data")
pdd = PatientDataDispatcher("config/config.yaml", dmo_features, MileStone.T2)
ids = list(set(pdd.metadata["Local.Participant"].to_list()))
dmo_data, dmo_labels = pdd.get_patient_data(PatientDataType.DMO, ids=ids)

config = lstm_regression
config.notes = "Regression with downsampling"

dmo_data_transform = Compose([Transform.center_dmo_data])
dmo_label_transform = Compose([Transform.normalise_dmo_label])

print("loading into dataloaders")
transforms = (dmo_data_transform, dmo_label_transform)
train, validation, test = dmo_into_dataloader(
    dmo_data, dmo_labels, config.batch_size, transforms, uniform_method=UniformMethod.UPSAMPLE
)

model = DMOLSTM(config).to(device=device)
optimiser = config.optimiser(model.parameters(), lr=config.learning_rate)

print("Beginning training")
lstm_train = LSTMRegressionTrain(
    model=model, optimiser=optimiser, device=device, config=config
)

lstm_train.train(train, validation, test)
