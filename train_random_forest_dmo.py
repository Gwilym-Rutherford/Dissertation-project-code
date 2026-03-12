# train classification with random forest for dmo data
from src.patient_data_dispatcher import PatientDataDispatcher, PatientDataType
from src.core.enums import MileStone, UniformMethod
from src.pipeline import dmo_for_random_forest
from src.model import DMORandomForest
from src.train import LSTMScaleTrain
from torchvision.transforms import Compose
from src.core.data_transforms import Transform

import torch

# for debuggin print matrix nicely with no line breaks
torch.set_printoptions(linewidth=200, sci_mode=False, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# print(dmo_data[0])
# quit()

# dmo_data_transform = [Transform.mask_dmo_data]
dmo_data_transform = [Transform.average_non_missing]
dmo_label_transform = [Transform.catagorise_dmo_label]

print("processing data")
transforms = (dmo_data_transform, dmo_label_transform)
train, test = dmo_for_random_forest(
    dmo_data,
    dmo_labels,
    transforms,
)

train_data, train_label = torch.Tensor.numpy(train[0]), torch.Tensor.numpy(train[1])
test_data, test_label = torch.Tensor.numpy(test[0]), torch.Tensor.numpy(test[1])

n_trees = 500
criterion = "gini"


random_forest = DMORandomForest(n_trees=n_trees, critirion=criterion)

random_forest.train(train_data, train_label)

print(random_forest.score(test_data, test_label))

# model = DMOLSTM(config).to(device=device)
# optimiser = config.optimiser(model.parameters(), lr=config.learning_rate)

# print("Beginning training")
# lstm_train = LSTMScaleTrain(
#     model=model, optimiser=optimiser, device=device, config=config
# )

# lstm_train.train(train, validation, test)
