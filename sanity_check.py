from src.model import DMOLSTM
from src.model import sanity
from src.logger import ExperimentLogger
from src.evaluation import Evaluation

import pandas as pd
import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(
    os.path.join(os.getcwd(), "data/sanity_check/household_power_consumption.txt"),
    sep=";",
    low_memory=False,
)

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

numerical_features = df.iloc[:, 2:]
features = numerical_features.to_numpy(dtype=np.float32)[:10000]

WINDOW_SIZE = 5

train_index = np.floor(len(features) * 0.8).astype(np.int32)

logger = ExperimentLogger(sanity)

training_data = torch.tensor(features[:train_index, :]).to(device=device)
test_data = torch.tensor(features[train_index:, :]).to(device=device)

model = DMOLSTM(sanity).to(device=device)
optimiser = sanity.optimiser(model.parameters(), lr=sanity.learning_rate)

for epoch in range(sanity.epochs):
    # train loop
    model.train()
    avg_loss = []
    for window in range(len(training_data) - WINDOW_SIZE - 1):
        data_in_window = training_data[window:window + WINDOW_SIZE, :]
        label = training_data[window + WINDOW_SIZE + 1, :]
        
        data_in_window = torch.tensor(data_in_window).unsqueeze(dim=0)
        label = torch.tensor(label)

        optimiser.zero_grad()
        pred = model(data_in_window).squeeze(dim=0)

        loss = sanity.loss_fn(pred, label)
        avg_loss.append(loss.item())
        
        loss.backward()
        optimiser.step()

    avg_loss = np.array(avg_loss)
    avg_loss_val = np.mean(avg_loss)
    # print(avg_loss_val)
    logger.log_values([("training_loss", avg_loss_val)])

    # validation loop
    model.eval()
    validation_avg_loss = []
    with torch.no_grad():
        for window in range(len(test_data) - WINDOW_SIZE - 1):
            data_in_window = training_data[window:window + WINDOW_SIZE, :]
            label = training_data[window + WINDOW_SIZE + 1, :]
            
            data_in_window = torch.tensor(data_in_window).unsqueeze(dim=0)
            label = torch.tensor(label)

            optimiser.zero_grad()
            pred = model(data_in_window).squeeze(dim=0)

            loss = sanity.loss_fn(pred, label)
            validation_avg_loss.append(loss.item())

        validation_avg_loss = np.array(validation_avg_loss)
        validation_loss_val = np.mean(validation_avg_loss)
        print(validation_loss_val)
        logger.log_values([("validation_loss", validation_loss_val)])


# final test loop
model.eval()
pred_list = []
labels_list = []
with torch.no_grad():
    for window in range(len(test_data) - WINDOW_SIZE - 1):
        data_in_window = training_data[window:window + WINDOW_SIZE, :]
        label = training_data[window + WINDOW_SIZE + 1, :]
        
        data_in_window = torch.tensor(data_in_window).unsqueeze(dim=0)
        label = torch.tensor(label)

        optimiser.zero_grad()
        pred = model(data_in_window).squeeze(dim=0)

        pred_list.append(pred)
        labels_list.append(label)

    pred = torch.cat(pred_list).cpu()
    labels = torch.cat(labels_list).cpu()

    evaluation = Evaluation(pred, labels)

    logger.save(evaluation, show_fig=False)
    
