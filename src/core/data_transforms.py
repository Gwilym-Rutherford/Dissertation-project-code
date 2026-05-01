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
    # prepends missing rows
    def fix_missing_dmo_data(dmo_data: DataFrame) -> DataFrame:
        dmo_data = dmo_data.head(7)
        dmo_data.fillna(MASK_VALUE, inplace=True)
        expected_n_rows = len(Day)

        n_missing_rows = expected_n_rows - dmo_data.shape[0]

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
        if len(data.shape) < 4:
            data = data.unsqueeze(dim=0)

        self.scaler = MinMaxScaler()

        patients, visit, day, features = data.shape
        data_2d = data.reshape(patients * visit * day, features)
        data_2d_scaled = self.scaler.fit_transform(data_2d)
        data = data_2d_scaled.reshape(patients, visit, day, features)

        data = torch.squeeze(torch.from_numpy(data), dim=0)
        return data

    def transform_dmo_data(self, data: DMOTensor):
        if len(data.shape) < 4:
            data = data.unsqueeze(dim=0)

        patient, visit, day, features = data.shape
        data_2d = data.reshape(patient * visit * day, features)
        data_2d_scaled = self.scaler.transform(data_2d)
        data = data_2d_scaled.reshape(patient, visit, day, features)
        
        data = data.squeeze()

        return data

    def fit_transform_dmo_labels(self, label: DMOTensor):
        self.scaler = MinMaxScaler()

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

    def fit_standard_scaler(self, data):
        self.scaler = StandardScaler()

        patients, visits, features = data.shape
        data_2d = data.reshape(patients * visits, features)
        self.scaler.fit(data_2d)

    def transform_standard_scaler(self, data):
        patients, visits, features = data.shape
        data_2d = data.reshape(patients * visits, features)
        data_2d_transformed = self.scaler.transform(data_2d)

        return torch.from_numpy(
                data_2d_transformed.reshape(patients, visits, features)
            )

    @staticmethod
    def get_patient_visits(dmo_data, dmo_labels, n_visits):
        feat_valid = (dmo_data != -1).all(dim=-1).all(dim=-1)
        label_valid = (dmo_labels.squeeze(-1) != -1)

        combined_valid = feat_valid & label_valid

        visit_counts = combined_valid.sum(dim=1)
        patient_indices = (visit_counts >= n_visits).nonzero(as_tuple=True)[0]

        if len(patient_indices) == 0:
            return torch.empty(0), torch.empty(0)

        final_feats = []
        final_labels = []

        for idx in patient_indices:
            mask = combined_valid[idx] 
            
            valid_feats = dmo_data[idx][mask][:n_visits]
            valid_lbls = dmo_labels[idx][mask][:n_visits]

            final_feats.append(valid_feats)
            final_labels.append(valid_lbls)

        return torch.stack(final_feats), torch.stack(final_labels)