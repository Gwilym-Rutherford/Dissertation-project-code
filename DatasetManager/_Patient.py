from dataclasses import dataclass
from DatasetManager.helper.named_tuple_def import RawData

import torch
import pandas


@dataclass
class _Patient:
    id: int
    meta_data: pandas.DataFrame | None
    sensor_dmo_data: dict[str, torch.Tensor] | None
    sensor_raw_data: dict[str, dict[str, RawData]] | None
