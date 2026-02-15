from DatasetManager.types import DMOTensor
from DatasetManager.helper.constants import MASK_VALUE
    
import torch

class Transform:
    
    @staticmethod
    def dmo_scale(dmo_data: DMOTensor) -> DMOTensor | None:
        if dmo_data is None:
            return torch.tensor([0])


        real_values_index = dmo_data != MASK_VALUE
        real_values = dmo_data[real_values_index]
        mean = real_values.mean()
        std = real_values.std()

        dmo_data[real_values_index] = (dmo_data[real_values_index] - mean) / std

        return dmo_data
