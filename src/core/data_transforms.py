from .types import DMOTensor
from .enums import Day, MileStone
from pandas import DataFrame

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
        real_values_index = dmo_data != MASK_VALUE
        real_values = dmo_data[real_values_index]

        if real_values.numel() == 0:
            return dmo_data

        mean = real_values.mean()
        std = real_values.std(unbiased=False)

        # edge case: dmo_values has only one value or all values are the same
        if torch.isnan(std) or std < 1e-8:
            std = torch.tensor(1.0, device=dmo_data.device, dtype=dmo_data.dtype)
        
        dmo_data = dmo_data.clone()
        dmo_data[real_values_index] = (dmo_data[real_values_index] - mean) / std

        return dmo_data
        
    @staticmethod
    def normalise_dmo_label(dmo_label: DMOTensor) -> DMOTensor:
        max_score = 21
        return dmo_label / max_score

    @staticmethod
    def mask_dmo_data(dmo_data: DMOTensor) -> DMOTensor:
        mask_boolean = (dmo_data != MASK_VALUE).to(torch.float32)
        mask_concat = torch.concatenate((dmo_data, mask_boolean), dim=1)

        return mask_concat
