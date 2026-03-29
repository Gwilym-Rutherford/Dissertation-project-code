from src.patient_data_dispatcher import PatientDataDispatcher, PatientDataType
from src.model import DMOLSTM
from src.core.enums import MileStone
from src.model import lstm_regression
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.regression import R2Score

import matplotlib.pyplot as plt

import numpy as np
import torch

# for debuggin print matrix nicely with no line breaks
torch.set_printoptions(linewidth=200, sci_mode=False, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# dmo_features = [
#     "wb_all_sum_d",
#     "walkdur_all_sum_d",
#     "wbsteps_all_sum_d",
#     "wbdur_all_avg_d",
#     "wbdur_all_p90_d",
#     "wbdur_all_var_d",
#     "cadence_all_avg_d",
#     "strdur_all_avg_d",
#     "cadence_all_var_d",
#     "strdur_all_var_d",
#     "ws_1030_avg_d",
#     "strlen_1030_avg_d",
#     "wb_10_sum_d",
#     "ws_10_p90_d",
#     "wb_30_sum_d",
#     "ws_30_avg_d",
#     "strlen_30_avg_d",
#     "cadence_30_avg_d",
#     "strdur_30_avg_d",
#     "ws_30_p90_d",
#     "cadence_30_p90_d",
#     "ws_30_var_d",
#     "strlen_30_var_d",
#     "wb_60_sum_d",
#     "total_worn_h_d",
#     "total_worn_during_waking_h_d",
# ]

# dmo_features = [
#     "ws_30_avg_d",
#     "strdur_all_var_d",
#     "wbdur_all_p90_d",
#     "wbsteps_all_sum_d",
#     "total_worn_during_waking_h_d",
# ]
# dmo feature have been selected through variance without min max normalising
dmo_features = [
    "wbsteps_all_sum_d",
    "wb_all_sum_d",
    "wbdur_all_var_d",
    "wb_10_sum_d",
    "walkdur_all_sum_d",
]

# dmo features sorted by variance after min max normalising
dmo_features = [
    "cadence_30_p90_d",
    "cadence_30_avg_d",
    "ws_30_p90_d",
    "strlen_1030_avg_d",
    "cadence_all_avg_d",
]

print("getting data")
pdd = PatientDataDispatcher("config/config.yaml", dmo_features, MileStone.ALL)
ids = list(set(pdd.metadata["Local.Participant"].to_list()))

static_features = pdd.metadata[["Local.Participant", "age", "weight", "height"]]


static_features = static_features.groupby("Local.Participant", as_index=False).agg(
    {"age": "max", "weight": "mean", "height": "first"}
)

dmo_data, dmo_labels = pdd.get_patient_data(PatientDataType.MILESTONE, ids=ids)


# remove patients that don't have a full dataset
patient_indexs = []
patient, visit, day, features = dmo_data.shape
for p in range(patient):
    all_visits = True
    for v in range(visit):
        data = dmo_data[p, v]
        label = dmo_labels[p, v]
        if (data == -1.0).any() or label == -1.0:
            all_visits = False

    if all_visits:
        patient_indexs.append(p)

dmo_data = dmo_data[patient_indexs]
dmo_labels = dmo_labels[patient_indexs]

config = lstm_regression
config.notes = "None"

def format_input_data(input_data, label_data):
    if len(input_data.shape) < 4:
        input_data = input_data.unsqueeze(dim=0)

    patient, visit, day, features = input_data.shape

    formatted_input_data = torch.zeros((patient, visit, (day * features) + 1))

    for p in range(patient):
        for v in range(visit):
            if v == 0:
                label = torch.tensor([0])
            else:
                label =label_data[p, v - 1]

            features = torch.flatten(input_data[p, v])

            features_and_lagged_label = torch.concatenate((features, label))

            formatted_input_data[p, v] = features_and_lagged_label
    
    return formatted_input_data

# x validation loop
accuracy = []
predicted_r2 = []
label_r2 = []
n_patients, n_visit, n_day, n_features = dmo_data.shape
for patient_index in range(n_patients):
    print(f"X Validation {patient_index}")
    # split data
    train_indexes = [i for i in range(n_patients) if i != patient_index]
    train_data = dmo_data[train_indexes].squeeze(dim=0)
    train_label = dmo_labels[train_indexes].squeeze(dim=0)

    test_data = dmo_data[patient_index]
    test_label = dmo_labels[patient_index].unsqueeze(dim=0)

    # fit and transform scaler on training data only
    scaler = MinMaxScaler()
    patients, visit, day, features = train_data.shape
    train_data_2d = train_data.reshape(patients * visit * day, features)
    train_data_2d_scaled = scaler.fit_transform(train_data_2d)
    train_data = train_data_2d_scaled.reshape(patients, visit, day, features)

    # transform test data
    visit, day, features = test_data.shape
    test_data_2d = test_data.reshape(visit * day, features)
    test_data_2d_scaled = scaler.transform(test_data_2d)
    test_data = test_data_2d_scaled.reshape(visit, day, features)

    # fit and transform scaler on train labels
    patients, visit, value = train_label.shape
    train_label_2d = scaler.fit_transform(train_label.reshape(patients * visit, value))
    train_label = train_label_2d.reshape(patients, visit, value)

    # transform test label
    patients, visit, value = test_label.shape
    test_label_2d = scaler.transform(test_label.reshape(patients * visit, value))
    test_label = test_label_2d.reshape(patients, visit, value)

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)

    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)

    # format data
    train_data = format_input_data(train_data, train_label)
    test_data = format_input_data(test_data, test_label)

    # convert to dataloaders for batching
    training_dataset = TensorDataset(train_data, train_label)
    testing_dataset = TensorDataset(test_data, test_label)

    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size)
    testing_dataloader = DataLoader(testing_dataset, batch_size=config.batch_size)


    model = DMOLSTM(config).to(device=device)
    optimiser = config.optimiser(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        training_loss = []
        model.train()
        for data, label in training_dataloader:
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            label = label[:, 4, 0]


            optimiser.zero_grad()
            pred = model(data).squeeze()

            loss = config.loss_fn(pred, label)
            training_loss.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

    # testing loop
    testing_loss = []
    model.eval()
    predicted_value = []
    actual_value = []
    with torch.no_grad():
        for data, label in testing_dataloader:
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            label = label[:, 4, 0].cpu()
            pred = model(data).view(-1).cpu()

            predicted_r2.append(pred[0])
            label_r2.append(label[0])

            dist = abs(label[0] - pred[0])
            if dist < 0.1:
                accuracy.append(1)
            else:
                accuracy.append(0)
                
            loss = config.loss_fn(pred, label)
            testing_loss.append(loss.item())

    
    print(f"Testing loss: {np.average(testing_loss)}")
    
    

print(f"Accuracy: {np.sum(accuracy)/len(accuracy) * 100}")

predicted = torch.tensor(predicted_r2)
label = torch.tensor(label_r2)

r2_calc = R2Score()
r2 = r2_calc(predicted, label).item()
print(f"R^2 {r2}")

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(predicted_r2, label_r2, alpha=0.5, edgecolors="k")

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Perfect Prediction")

ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs. Predicted (Regression)")
ax.legend()
plt.show()