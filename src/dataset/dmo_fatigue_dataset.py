from torch.utils.data import Dataset
from src.core.types import DMOTensor


class DMOFatigueDataset(Dataset):
    def __init__(
        self,
        labels: DMOTensor,
        dmo_data: DMOTensor,
        transform=None,
        target_transform=None,
    ) -> None:
        self.labels = labels
        self.dmo_data = dmo_data
        self.transform = transform
        self.target_transform = target_transform

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
