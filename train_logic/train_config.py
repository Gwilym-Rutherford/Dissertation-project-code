from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class DataLoaderConfig():
    train_dataloader: DataLoader
    validation_dataloader: DataLoader
    test_dataloader: DataLoader