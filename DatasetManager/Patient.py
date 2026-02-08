from dataclasses import dataclass

import torch
import pandas


@dataclass
class Patient:
    id: int
    meta_data: pandas.DataFrame
    sensor_dmo_data: torch.tensor
    sensor_raw_data: torch.tensor
