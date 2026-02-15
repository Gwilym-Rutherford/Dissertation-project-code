from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Patient import Patient
    from .helper.named_tuple_def import RawData

import torch
import pandas

type Ids = torch.Tensor
type Patients = list[Patient]
type RawSensorData = dict[str, dict[str, RawData]]

type CSVData = pandas.DataFrame
type DMOFeatures = list[str] | None
type DMOTensor = torch.Tensor
