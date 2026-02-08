from typing import NamedTuple, Any

import torch

class Condition(NamedTuple):
    column: str
    operation: str
    value: Any

class RawData(NamedTuple):
    start_date_time: str
    time_zone: str
    fs: int | float
    time_stamp: torch.Tensor
    acc: torch.Tensor
    gyr: torch.Tensor
