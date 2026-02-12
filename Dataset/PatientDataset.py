from torch.utils.data import Dataset
from DatasetManager._Patient import _Patient
from typing import TypeAlias
from DatasetManager.helper.enum_def import MileStone, Day
from collections.abc import Callable

import torch

patients: TypeAlias = list[_Patient]


class FatigueDMODataset(Dataset):
    def __init__(
        self,
        patients: patients,
        milestone: MileStone,
        n_features: int,
        transform: Callable = None,
        target_transform: Callable = None,
    ):
        self.patients = patients
        self.milestone = milestone
        self.n_features = n_features
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        fatigue = self.patients[idx].get_fatigue_at_milestone(self.milestone)
        dmo_data = self.patients[idx].sensor_dmo_data

        if dmo_data is None:
            dmo_data = torch.zeros(
                (int(self.milestone.value[-1])) * len(Day), self.n_features
            )

        if self.transform:
            dmo_data = self.transform(dmo_data)
        if self.target_transform:
            fatigue = self.target_transform(fatigue)

        return dmo_data, fatigue
