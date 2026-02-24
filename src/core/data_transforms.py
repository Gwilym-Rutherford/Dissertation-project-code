from .types import DMOTensor
from .enums import Day, MileStone, UniformMethod
from pandas import DataFrame
from math import ceil

import numpy as np
import pandas as pd
import torch


MASK_VALUE = 0


class Transform:
    @staticmethod
    def fix_missing_dmo_data(dmo_data: DataFrame) -> DataFrame:
        dmo_data.fillna(MASK_VALUE, inplace=True)
        expected_shape = (len(Day), len(MileStone) - 1)

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

    @staticmethod
    def normalise_dmo_label(dmo_label: DMOTensor) -> DMOTensor:
        max_score = 84
        return dmo_label / max_score

    @staticmethod
    def mask_dmo_data(dmo_data: DMOTensor) -> DMOTensor:
        mask_boolean = (dmo_data != MASK_VALUE).to(torch.float32)
        mask_concat = torch.concatenate((dmo_data, mask_boolean), dim=1)

        return mask_concat

    @staticmethod
    def uniform_dmo(
        dmo_data: torch.Tensor, dmo_labels: torch.Tensor, method: UniformMethod
    ) -> tuple[torch.Tensor, torch.Tensor]:

        bins = 20
        threshold = 20

        min_val = torch.min(dmo_labels)
        max_val = torch.max(dmo_labels)

        bin_indices = ((dmo_labels - min_val) / (max_val - min_val) * bins).to(
            torch.long
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
                if len(bin_) < threshold:
                    missing = threshold - len(bin_)
                    arr_scalar = ceil(missing / len(bin_))
                    mask.extend([x for x in (bin_ * arr_scalar)[:missing]])

        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return dmo_data[mask_tensor], dmo_labels[mask_tensor]
