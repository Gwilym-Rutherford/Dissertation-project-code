from src.core.data_transforms import Transform
from src.core.types import DMOTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from src.dataset import DMOFatigueDataset
from ..split_data import split_data
from ..clean_data import clean_dmo_data
from src.research_util import plot_distribution

import torch


def dmo_into_dataloader(
    dmo_data: DMOTensor,
    dmo_labels: DMOTensor,
    batch_size: int,
    training: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    downsample_uniform: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    dmo_data, dmo_labels = clean_dmo_data(dmo_data, dmo_labels)

    if downsample_uniform:
        dmo_data, dmo_labels = Transform.uniform_downsample_dmo(dmo_data, dmo_labels)
        # plot_distribution(
        #     torch.Tensor.tolist(dmo_labels.to(dtype=torch.int8)),
        #     "testing_uniform_downsample_dmo",
        # )

    train_data, validation_data, test_data = split_data(
        dmo_data, training, validation, test
    )

    train_label, validation_label, test_label = split_data(
        dmo_labels, training, validation, test
    )

    # testing method from paper
    # dmo_data_transform = Compose(
    #     [Transform.center_dmo_data, Transform.mask_dmo_data]
    # )

    dmo_data_transform = Compose([Transform.center_dmo_data])

    dmo_label_transform = Compose([Transform.normalise_dmo_label])

    dataset_training = DMOFatigueDataset(
        train_label,
        train_data,
        transform=dmo_data_transform,
        target_transform=dmo_label_transform,
    )

    dataset_validation = DMOFatigueDataset(
        validation_label,
        validation_data,
        transform=dmo_data_transform,
        target_transform=dmo_label_transform,
    )

    dataset_testing = DMOFatigueDataset(
        test_label,
        test_data,
        transform=dmo_data_transform,
        target_transform=dmo_label_transform,
    )

    dataloader_training = DataLoader(dataset_training, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
    dataloader_testing = DataLoader(dataset_testing, batch_size=batch_size, shuffle=True)

    return (dataloader_training, dataloader_testing, dataloader_validation)
