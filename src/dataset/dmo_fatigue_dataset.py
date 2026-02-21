from torch.utils.data import Dataset
from src.core.types import DMOTensor

import torch


class DMOFatigueDataset(Dataset):
    def __init__(
        self,
        labels: DMOTensor,
        dmo_data: DMOTensor,
        transform=None,
        target_transform=None,
    ) -> None:
        #self.labels = labels
        #self.dmo_data = dmo_data
        self.transform = transform
        self.target_transform = target_transform


        label_mask = []
        for i_label, label in enumerate(labels):
            label = target_transform(label)
            if not torch.isnan(label):
                label_mask.append(i_label)

        dmo_mask = []
        for i_dmo_data, dmo in enumerate(dmo_data):
            dmo = transform(dmo)
            if not (dmo == 0).all():
                dmo_mask.append(i_dmo_data)

        mask = list(set(label_mask + dmo_mask))
        mask.sort()

        self.labels = labels[mask]
        self.dmo_data = dmo_data[mask]



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        dmo_data = self.dmo_data[idx]

        if self.transform:
            dmo_data = self.transform(dmo_data)
        if self.target_transform:
            label = self.target_transform(label)

        return dmo_data, label
