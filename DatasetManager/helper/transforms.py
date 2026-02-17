from DatasetManager.types import DMOTensor
from DatasetManager.helper.constants import MASK_VALUE
    
import torch

class Transform:
    
    @staticmethod
    def dmo_normalise(dmo_data: DMOTensor) -> DMOTensor:
        if torch.equal(torch.tensor([-1]), dmo_data):
            return torch.tensor([-1])

        real_values_index = dmo_data != MASK_VALUE
        real_values = dmo_data[real_values_index]
        mean = real_values.mean()
        std = real_values.std()

        dmo_data[real_values_index] = (dmo_data[real_values_index] - mean) / std
        
        return dmo_data


    @staticmethod
    def mask_data(dmo_data: DMOTensor) -> DMOTensor:
        if torch.equal(torch.tensor([-1]), dmo_data):
            return torch.tensor([-1])
            
        mask_boolean = (dmo_data != MASK_VALUE).to(torch.float32)
        mask_concat = torch.concatenate((dmo_data, mask_boolean), dim=1)

        return mask_concat