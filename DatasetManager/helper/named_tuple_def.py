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


class SplitData(NamedTuple):
    training: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor


class SplitRatio(NamedTuple):
    training: float
    validation: float
    test: float

    def sum(self) -> float:
        return self.training + self.validation + self.test
