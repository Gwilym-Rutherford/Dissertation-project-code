from typing import TypeAlias
from .Patient import Patient

import torch
import pandas

Ids: TypeAlias = torch.Tensor
Patients: TypeAlias = dict[int, Patient]

CSVData: TypeAlias = pandas.DataFrame
DMOFeatures: TypeAlias = list[str] | None