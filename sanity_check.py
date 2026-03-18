from src.model import DMOLSTM
from src.model import sanity
from src.logger import ExperimentLogger
from src.evaluation import Evaluation
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
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
    label_data.append(day_tensor[-1])

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
testing_input = torch.tensor(testing_input_2d).view(x, y, z)

# Remove all input and respected label if the label == NaN
