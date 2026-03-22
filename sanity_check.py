from src.model import DMOLSTM
from src.model import sanity
from src.logger import ExperimentLogger
from src.evaluation import Evaluation
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import IterativeImputer

import pandas as pd
import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import dataset
df = pd.read_csv(
    os.path.join(os.getcwd(), "data/sanity_check/beijing_sanity_check.csv"),
    sep=",",
    low_memory=False,
)
df = df.drop(columns=["Iws", "Is", "Ir"])

# wind direction encoding
encoder = LabelEncoder()
df["cbwd"] = encoder.fit_transform(df["cbwd"])

# split into days and data to fit impute
days = {}

for index, row in df.iterrows():
    year_month_day = f"{row['year']}{row['month']}{row['day']}"
    if year_month_day not in days:
        days[year_month_day] = []

    days[year_month_day].append(
        torch.tensor([row["pm2.5"], row["DEWP"], row["TEMP"], row["PRES"], row["cbwd"]])
    )

# split into input and label data
input_data = []
label_data = []

for _, value in days.items():
    day_tensor = torch.stack(value)

    input_data.append(day_tensor[:-1])
    label_data.append(day_tensor[-1][0])

input_data = torch.stack(input_data)
label_data = torch.stack(label_data)

# split data into training and testing data
training_split = 0.8
testing_split = 0.2

n_samples = len(label_data)
training_index = np.floor(n_samples * training_split).astype(np.int64)

training_input = input_data[0:training_index]
training_label = label_data[0:training_index]

testing_input = input_data[training_index:]
testing_label = label_data[training_index:]

# impute missing data only fitting on training data
impute = IterativeImputer(random_state=0, missing_values=np.nan)
x, y, z = training_input.shape
train_input_2d = torch.reshape(training_input, (x * y, z))
train_imputed_2d = impute.fit_transform(train_input_2d)
training_input = torch.tensor(train_imputed_2d).view(x, y, z)

x, y, z = testing_input.shape
testing_input_2d = torch.reshape(testing_input, (x * y, z))
testing_imputed_2d = impute.transform(testing_input_2d)
testing_input = torch.tensor(testing_imputed_2d).view(x, y, z)

# Remove all input and respected label if the label == NaN
mask_training = ~torch.isnan(training_label)
training_input = training_input[mask_training]
training_label = training_label[mask_training]

mask_testing = ~torch.isnan(testing_label)
testing_input = testing_input[mask_testing]
testing_label = testing_label[mask_testing]

# scale values 0-1
training_min = training_input.amin(dim=(0, 1), keepdim=True)
training_max = training_input.amax(dim=(0, 1), keepdim=True)
training_input = (training_input - training_min) / (training_max - training_min + 1e-8)

testing_min = testing_input.amin(dim=(0, 1), keepdim=True)
testing_max = testing_input.amax(dim=(0, 1), keepdim=True)
testing_input = (testing_input - training_min) / (training_max - training_min + 1e-8)

target_scaler = MinMaxScaler(feature_range=(0,1))

training_label = torch.tensor(target_scaler.fit_transform(training_label.reshape(-1, 1))).squeeze()
testing_label = torch.tensor(target_scaler.transform(testing_label.reshape(-1, 1))).squeeze()

# turn to dataloaders so i can batch it
training_dataset = TensorDataset(training_input, training_label)
testing_dataset = TensorDataset(testing_input, testing_label)

training_dataloader = DataLoader(training_dataset, batch_size=16)
testing_dataloader = DataLoader(testing_dataset, batch_size=16)

# model setup
model = DMOLSTM(sanity).to(device=device)
optimiser = sanity.optimiser(model.parameters(), lr=sanity.learning_rate)

logger = ExperimentLogger(sanity)

# training loop
for epoch in range(sanity.epochs):
    training_loss = []
    model.train()
    for data, label in training_dataloader:
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        optimiser.zero_grad()
        pred = model(data).squeeze()

        loss = sanity.loss_fn(pred, label)
        training_loss.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
    
    avg_loss = np.average(training_loss)
    print(avg_loss)
    logger.log_values([("training_loss", avg_loss)])

# testing loop
print("testing now")
testing_loss = []
predications = []
labels = []
model.eval()
with torch.no_grad():
    for data, label in testing_dataloader:
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        pred = model(data).squeeze()
        
        for pred, label in zip(pred, label):
            predications.append(pred.item())
            labels.append(label.item())

            logger.log_values([("pred - actual",label.item() - pred.item())])

        loss = sanity.loss_fn(pred, label)
        testing_loss.append(loss.item())

    print(np.average(testing_loss))

    predications = torch.tensor(predications)
    label = torch.tensor(labels)

    evaluation = Evaluation(predications, labels, scale=False)

    logger.save(evaluation, show_fig=True)


