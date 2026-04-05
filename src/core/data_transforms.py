from .types import DMOTensor
from typing import Literal
from .enums import Day, UniformMethod
from pandas import DataFrame
from math import ceil

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import torch


MASK_VALUE = -1.0


class Transform:
    @staticmethod
    def fix_missing_dmo_data(dmo_data: DataFrame) -> DataFrame:
        dmo_data.fillna(MASK_VALUE, inplace=True)
        number_of_milestones = 5
        expected_shape = (len(Day), number_of_milestones)

        if dmo_data.shape == expected_shape:
            return dmo_data

        n_missing_rows = expected_shape[0] - dmo_data.shape[0]

        if n_missing_rows > 0:
            padding = pd.DataFrame(
                MASK_VALUE, index=np.arange(n_missing_rows), columns=dmo_data.columns
            )
            dmo_data = pd.concat([padding, dmo_data], axis=0)

        return dmo_data

    @staticmethod
    def uniform_dmo(
        dmo_data: torch.Tensor, dmo_labels: torch.Tensor, method: UniformMethod
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # These are somehwat arbitrary, and can be tweaked
        bins = 20
        threshold = 100

        min_val = torch.min(dmo_labels)
        max_val = torch.max(dmo_labels)

        bin_indices = ((dmo_labels - min_val) / (max_val - min_val + 1e-8) * bins).to(
            dtype=torch.int8
        )

        bin_counts = torch.zeros(bins, dtype=torch.int8)
        mask = []
        bin_members = [[] for _ in range(bins)]

        for i, bin_id in enumerate(bin_indices):
            id_ = bin_id.item() - 1
            if bin_counts[id_] < threshold:
                bin_counts[id_] += 1
                bin_members[id_].append(i)
                mask.append(i)

        if method == UniformMethod.UPSAMPLE:
            for bin_ in bin_members:
                if len(bin_) < threshold and len(bin_) != 0:
                    missing = threshold - len(bin_)
                    arr_scalar = ceil(missing / len(bin_))
                    mask.extend([x for x in (bin_ * arr_scalar)[:missing]])

        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return dmo_data[mask_tensor], dmo_labels[mask_tensor]

    @staticmethod
    def format_label_data(label_data: DMOTensor):
        patient, visit, label = label_data.shape
        label_data = label_data.reshape(patient * visit, label)
        return label_data

    @staticmethod
    def format_input_data(input_data: DMOTensor):
        if len(input_data.shape) < 4:
            input_data = input_data.unsqueeze(dim=0)

        patient, visit, day, features = input_data.shape
        input_data = input_data.reshape(patient * visit, day, features)

        return input_data

    @staticmethod
    def format_input_data_delta_day(input_data: DMOTensor):
        if len(input_data.shape) < 4:
            input_data = input_data.unsqueeze(dim=0)

        patients, visits, days, features = input_data.shape

        delta_features = torch.zeros(patients, visits, days - 1, features)

        for patient in range(patients):
            for visit in range(visits):
                day_feature = input_data[patient, visit]

                reference_features = day_feature[0, :]
                for day in range(days - 1):
                    updated_feature = day_feature[day + 1, :] - reference_features
                    delta_features[patient, visit, day] = updated_feature

        input_data = delta_features.reshape(patients * visits, days - 1, features)

        return input_data

    @staticmethod
    def format_input_data_delta_visit(input_data: DMOTensor):
        if len(input_data.shape) < 4:
            input_data = input_data.unsqueeze(dim=0)

        patients, visits, days, features = input_data.shape

        delta_visits = torch.zeros(patients, visits - 1, days, features)
        for patient in range(patients):
            reference_visit = input_data[patient, 0]
            for visit in range(visits - 1):
                updated_visit = input_data[patient, visit + 1] - reference_visit
                delta_visits[patient, visit] = updated_visit

        input_data = delta_visits.reshape(patients * (visits - 1), days, features)

        return input_data

    @staticmethod
    def format_label_data_delta_visit(label_data: DMOTensor):
        patients, visits, labels = label_data.shape

        delta_labels = torch.zeros(patients * (visits - 1), labels)
        for patient in range(patients):
            reference_label = label_data[patient, 0]

            for visit in range(visits - 1):
                updated_label = label_data[patient, visit + 1] - reference_label
                delta_labels[patient] = updated_label

        label_data = delta_labels.reshape(patients * (visits - 1), labels)
        return label_data

    def fit_transform_dmo_data(self, data: DMOTensor):
        self.scaler = MinMaxScaler()
        # self.scaler = StandardScaler()

        patients, visit, day, features = data.shape
        data_2d = data.reshape(patients * visit * day, features)
        data_2d_scaled = self.scaler.fit_transform(data_2d)
        data = data_2d_scaled.reshape(patients, visit, day, features)

        return data

    def transform_dmo_data(self, data: DMOTensor):
        if len(data.shape) > 3:
            data = data.squeeze(dim=0)

        visit, day, features = data.shape
        data_2d = data.reshape(visit * day, features)
        data_2d_scaled = self.scaler.transform(data_2d)
        data = data_2d_scaled.reshape(visit, day, features)

        return data

    def fit_transform_dmo_labels(self, label: DMOTensor):
        self.scaler = MinMaxScaler()
        # self.scaler = StandardScaler()

        patients, visit, value = label.shape
        label_2d = self.scaler.fit_transform(label.reshape(patients * visit, value))
        label = label_2d.reshape(patients, visit, value)

        return label

    def transform_dmo_labels(self, label: DMOTensor):
        patients, visit, value = label.shape
        label_2d = self.scaler.transform(label.reshape(patients * visit, value))
        label = label_2d.reshape(patients, visit, value)

        return label

    @staticmethod
    def format_input_data_time_seq_days(input_data: DMOTensor):
        if len(input_data.shape) < 4:
            input_data = input_data.unsqueeze(dim=0)

        patient, visit, day, features = input_data.shape
        input_data = input_data.reshape(patient * visit, day, features)

        return input_data

    @staticmethod
    def format_label_time_seq_visit(label_data: DMOTensor):
        patient, visit, label = label_data.shape
        label_data = label_data.reshape(patient * visit, label)
        return label_data

    @staticmethod
    def data_to_dataloaders(data: DMOTensor, label: DMOTensor, **kwargs):
        dataset = TensorDataset(data, label)
        return DataLoader(dataset, **kwargs)

    @staticmethod
    def format_input_data_lag_label(input_data, label_data):
        if len(input_data.shape) < 4:
            input_data = input_data.unsqueeze(dim=0)

        patient, visit, day, features = input_data.shape
        formatted_input_data = torch.zeros((patient, visit, (day * features) + 1))

        for p in range(patient):
            for v in range(visit):
                if v == 0:
                    label = torch.tensor([0])
                else:
                    label = label_data[p, v - 1]
                features = torch.flatten(input_data[p, v])
                features_and_lagged_label = torch.concatenate((features, label))
                formatted_input_data[p, v] = features_and_lagged_label

        return formatted_input_data

    @staticmethod
    def format_input_data_only_label(label_data):
        if len(label_data.shape) < 3:
            label_data = label_data.unsqueeze(dim=0)

        patients, visits, label = label_data.shape

        formatted_input_data = torch.zeros_like(label_data)

        for p in range(patients):
            visit = label_data[p]

            updated_input_data = torch.concatenate((torch.tensor([[0]]), visit[:-1]))

            formatted_input_data[p] = updated_input_data

        return formatted_input_data
