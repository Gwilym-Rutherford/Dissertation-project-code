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
    training: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    uniform_method: UniformMethod = None,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    # remove nan lables and associated input
    dmo_data, dmo_labels = Transform.clean_dmo_data(dmo_data, dmo_labels)
    
    # split data into training, validation and test sets
    train_data, validation_data, test_data = split_data(
        dmo_data, training, validation, test
    )

    # split labels into training, validation and test sets
    train_label, validation_label, test_label = split_data(
        dmo_labels, training, validation, test
    )

    # scale input to the range 0-1
    train_data, validation_data, test_data = Transform.min_max_scale_input_data(
        train_data, validation_data, test_data
    )

    # sclale label to the range 0-1
    train_label, validation_label, test_label = Transform.normalise_dmo_label(
        train_label, validation_label, test_label
    )

    # fit impute model on training data only
    Transform.fit_impute_dmo_data(train_data)

    if uniform_method is not None:
        dmo_data, dmo_labels = Transform.uniform_dmo(
            train_data, train_label, uniform_method
        )
        # plot_distribution(
        #     torch.Tensor.tolist(dmo_labels.to(dtype=torch.int8)),
        #     "testing_uniform_upsample_dmo",
        # )
        # quit()

    train_data = Transform.imput_dmo_data(train_data)
    validation_data = Transform.imput_dmo_data(validation_data)
    test_data = Transform.imput_dmo_data(test_data)


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

    dataloader_training = DataLoader(
        dataset_training, batch_size=batch_size, shuffle=True
    )
    dataloader_validation = DataLoader(
        dataset_validation, batch_size=batch_size, shuffle=True
    )
    dataloader_testing = DataLoader(
        dataset_testing, batch_size=batch_size, shuffle=True
    )

    return (dataloader_training, dataloader_testing, dataloader_validation)
