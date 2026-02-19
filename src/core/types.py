from __future__ import annotations
from typing import TYPE_CHECKING
from .enums import MileStone

if TYPE_CHECKING:
    pass

import torch
import pandas

type Ids = torch.Tensor

type CSVData = pandas.DataFrame
type DMOFeatures = list[str] | None
type DMOTensor = torch.Tensor

type PatientData = CSVData
type ListIds = list[int] | set[int] | None

