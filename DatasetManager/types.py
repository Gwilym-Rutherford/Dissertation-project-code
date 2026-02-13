from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Patient import Patient

import torch
import pandas

type Ids = torch.Tensor
type Patients = dict[int, Patient]

type CSVData = pandas.DataFrame
type DMOFeatures = list[str] | None
