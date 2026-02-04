import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class FatigueDataset(Dataset):
    def __init__(
        self,
        fatigue_questionaire,
        fatigue_results,
        transform=None,
        target_transform=None,
    ):
        self.fatigue_questionaire = fatigue_questionaire
        self.fatigue_results = fatigue_results
        self.transform = transform
        self. target_transform = target_transform


    def __len__(self):
        return len(self.fatigue_results)

    def __getitem__(self, idx):
        questionaire = self.questionaire[idx]
        result = self.fatigue_results[idx]

        if self.transform:
            questionaire = self.transform(questionaire)
        if self.target_transform:
            result = self.target_transform(result) 

        return questionaire, result
