from .types import DMOTensor
from typing import Literal
from .enums import Day, UniformMethod
from pandas import DataFrame
from math import ceil

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

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
    def normalise_dmo_data(dmo_data: DMOTensor) -> DMOTensor:
        n_rows, _ = dmo_data.shape

        for row in range(n_rows):
            day_tensor = dmo_data[row, :]
            real_values_index = day_tensor != MASK_VALUE
            real_values = day_tensor[real_values_index]

            if torch.numel(real_values) == 0:
                continue

            min_value = torch.min(real_values)
            max_value = torch.max(real_values)
            diff = max_value - min_value

            if diff != 0:
                normalised_tensor = (day_tensor[real_values_index] - min_value) / (
                    max_value - min_value
                )
            else:
                normalised_tensor = day_tensor[real_values_index] = 0

            dmo_data[row, real_values_index] = normalised_tensor

        return dmo_data

    @staticmethod
    def min_max_scale_input_data(
        training_input: torch.Tensor,
        validation_input: torch.Tensor,
        testing_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        train_mask = training_input != MASK_VALUE
        t_min = training_input.masked_fill(~train_mask, float("inf")).amin(
            dim=(0, 1), keepdim=True
        )
        t_max = training_input.masked_fill(~train_mask, float("-inf")).amax(
            dim=(0, 1), keepdim=True
        )

        denom = t_max - t_min + 1e-8

        def apply_scaling(data):
            mask = data != MASK_VALUE
            scaled_data = (data - t_min) / denom
            return torch.where(mask, scaled_data, data)

        return (
            apply_scaling(training_input),
            apply_scaling(validation_input),
            apply_scaling(testing_input),
        )

    @staticmethod
    def center_dmo_data(dmo_data: DMOTensor) -> DMOTensor:
        n_rows, _ = dmo_data.shape

        for row in range(n_rows):
            day_tensor = dmo_data[row, :]
            real_values_index = day_tensor != MASK_VALUE
            real_values = day_tensor[real_values_index]

            if torch.numel(real_values) == 0:
                continue

            mean = torch.mean(real_values)
            std = torch.std(real_values)

            normalised_tensor = (day_tensor[real_values_index] - mean) / std

            dmo_data[row, real_values_index] = normalised_tensor

        return dmo_data

    @classmethod
    def fit_impute_dmo_data(cls, dmo_data: DMOTensor) -> DMOTensor:
        cls.impute = IterativeImputer(missing_values=MASK_VALUE, tol=1e-2, max_iter=100)
        patient, visits, day, features = dmo_data.shape
        dmo_data_input_2d = dmo_data.reshape(patient * visits * day, features)
        torch.nan_to_num_(dmo_data_input_2d, MASK_VALUE)
        cls.impute.fit(dmo_data_input_2d)

    @classmethod
    def imput_dmo_data(cls, dmo_data: DMOTensor) -> DMOTensor:
        patients, visits, day, features = dmo_data.shape

        for patient in range(patients):
            patient_mat = dmo_data[patient]
            x, y, z = patient_mat.shape
            patient_mat = patient_mat.reshape(x * y, z)
            patient_imputed = cls.impute.transform(patient_mat)
            dmo_data[patient] = torch.tensor(patient_imputed.reshape(x, y, z))

        return dmo_data


    @staticmethod
    def normalise_dmo_label(
        training_labels,
        validation_labels,
        testing_labels: DMOTensor,
        feature_range=(0, 1),
    ) -> tuple[DMOTensor, DMOTensor, DMOTensor]:
        scaler = MinMaxScaler(feature_range=feature_range)

        training_labels = torch.tensor(
            scaler.fit_transform(training_labels.reshape(-1, 1))
        ).squeeze()
        validation_labels = torch.tensor(
            scaler.fit_transform(validation_labels.reshape(-1, 1))
        ).squeeze()
        testing_labels = torch.tensor(
            scaler.fit_transform(testing_labels.reshape(-1, 1))
        ).squeeze()

        return training_labels, validation_labels, testing_labels

    @staticmethod
    def catagorise_dmo_label(dmo_label: DMOTensor) -> DMOTensor:
        max_theoretical_value = 21 * 5
        # if changing this make sure to change the output size for lstm_scale
        n_catagories = 10

        catagory_width = max_theoretical_value / n_catagories
        catagories = (dmo_label / catagory_width).floor().long()
        return catagories

    @staticmethod
    def clean_dmo_data(
        dmo_data: DMOTensor, labels: DMOTensor
    ) -> tuple[DMOTensor, DMOTensor]:
        label_nan_mask = ~torch.isnan(labels)
        empty_dmo_data_mask = dmo_data.sum(dim=(1, 2)) != 0

        mask = label_nan_mask & empty_dmo_data_mask

        labels = labels[mask]
        dmo_data = dmo_data[mask]

        return dmo_data, labels

    @staticmethod
    # should be dim = 1 for regular training, but 2 for random forest
    def mask_dmo_data(dmo_data: DMOTensor, dim: int = 1) -> DMOTensor:
        mask_boolean = (dmo_data != MASK_VALUE).to(torch.float32)
        mask_concat = torch.concatenate((dmo_data, mask_boolean), dim=2)

        return mask_concat

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
    def scale_output_to_single_value(scale_output: torch.Tensor) -> torch.Tensor:
        values, indices = torch.max(scale_output, dim=1)
        return values

    @staticmethod
    def average_non_missing(data: torch.Tensor) -> torch.Tensor:
        mask = (data != MASK_VALUE).to(torch.long)

        clean_data = data * mask
        sum_vals = torch.sum(clean_data, dim=1)
        counts = torch.sum(mask, dim=1)
        counts = torch.clamp(counts, min=1)
        return sum_vals / counts

