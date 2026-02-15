from torch.utils.data import Dataset
from DatasetManager.helper.enum_def import MileStone, Day
from DatasetManager.helper.constants import MASK_VALUE
from collections.abc import Callable
from DatasetManager.types import Patients



class FatigueDMODataset(Dataset):
    def __init__(
        self,
        patients: Patients,
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
        
        if self.transform:
            dmo_data = self.transform(dmo_data)
        if self.target_transform:
            fatigue = self.target_transform(fatigue)

        return dmo_data, fatigue
