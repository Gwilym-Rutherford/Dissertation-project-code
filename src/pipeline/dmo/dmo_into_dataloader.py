from src.core.data_transforms import Transform
from src.core.types import DMOTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from src.dataset import DMOFatigueDataset
from ..split_data import split_data
from src.research_util import plot_distribution
from src.core.enums import UniformMethod

import torch


def dmo_into_dataloader(
    dmo_data: DMOTensor,
    dmo_labels: DMOTensor,
    batch_size: int,
    transforms: tuple[Compose, Compose],
    training: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    uniform_method: UniformMethod = None,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    dmo_data, dmo_labels = Transform.clean_dmo_data(dmo_data, dmo_labels)

    if shuffle:
        perm_mask = torch.randperm(dmo_data.shape[0])
        dmo_data = dmo_data[perm_mask]
        dmo_labels = dmo_labels[perm_mask]

    train_data, validation_data, test_data = split_data(
        dmo_data, training, validation, test
    )

    train_label, validation_label, test_label = split_data(
        dmo_labels, training, validation, test
    )

    if uniform_method is not None:
        dmo_data, dmo_labels = Transform.uniform_dmo(train_data, train_label, uniform_method)
        # plot_distribution(
        #     torch.Tensor.tolist(dmo_labels.to(dtype=torch.int8)),
        #     "testing_uniform_upsample_dmo",
        # )
        # quit()

    dmo_data_transform, dmo_label_transform = transforms

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

    dataloader_training = DataLoader(dataset_training, batch_size=batch_size, shuffle=False)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
    dataloader_testing = DataLoader(dataset_testing, batch_size=batch_size, shuffle=False)

    return (dataloader_training, dataloader_testing, dataloader_validation)
