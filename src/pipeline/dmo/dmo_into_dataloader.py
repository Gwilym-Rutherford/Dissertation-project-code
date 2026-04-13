from src.core.data_transforms import Transform
from src.core.types import DMOTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from src.dataset import DMOFatigueDataset
from ..split_data import split_data
from src.research_util import plot_distribution
from src.core.enums import UniformMethod

import numpy as np
import torch


def dmo_into_dataloader(
    dmo_data: DMOTensor,
    dmo_labels: DMOTensor,
    batch_size: int,
    transforms: tuple[Compose, Compose],
    training: float = 0.8,
    # validation: float = 0,
    test: float = 0.2,
    uniform_method: UniformMethod = None,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    
    # min max normalise labels globally
    transform = Transform()
    dmo_labels = transform.fit_transform_dmo_data(dmo_labels)

    # split data into training, validation and test sets
    train_data, validation_data, test_data = split_data(
        dmo_data, training, 0, test
    )

    # split labels into training, validation and test sets
    train_label, validation_label, test_label = split_data(
        dmo_labels, training, 0, test
    )

    # fit standard scaler on training data only
    transform.fit_standard_scaler(train_data)
    train_data = transform.transform_standard_scaler(train_data)

    # apply fit to testing data
    test_data = transform.transform_standard_scaler(test_data)
    
    dmo_data_transform, dmo_label_transform = transforms

    dataset_training = DMOFatigueDataset(
        train_label,
        train_data,
        transform=dmo_data_transform,
        target_transform=dmo_label_transform,
    )

    # dataset_validation = DMOFatigueDataset(
    #     validation_label,
    #     validation_data,
    #     transform=dmo_data_transform,
    #     target_transform=dmo_label_transform,
    # )

    dataset_testing = DMOFatigueDataset(
        test_label,
        test_data,
        transform=dmo_data_transform,
        target_transform=dmo_label_transform,
    )

    dataloader_training = DataLoader(
        dataset_training, batch_size=batch_size, shuffle=True
    )
    
    # dataloader_validation = DataLoader(
    #     dataset_validation, batch_size=batch_size, shuffle=True
    # )

    dataloader_testing = DataLoader(
        dataset_testing, batch_size=batch_size, shuffle=True
    )

    return (dataloader_training, dataloader_testing)
