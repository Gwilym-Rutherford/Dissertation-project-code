from src.core.data_transforms import Transform
from src.core.types import DMOTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from ..split_data import split_data

import torch



def dmo_for_random_forest(
    dmo_data: DMOTensor,
    dmo_labels: DMOTensor,
    transforms: tuple[list[callable], list[callable]],
    normalise: bool = True,
    training: float = 0.8,
    # validation set not needed
    validation: float = 0,
    test: float = 0.20,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:

    if normalise:
        # min max normalise labels globally
        transform = Transform()
        dmo_labels = transform.fit_transform_dmo_data(dmo_labels)


    train_data, validation_data, test_data = split_data(
        dmo_data, training, validation, test
    )

    train_label, validation_label, test_label = split_data(
        dmo_labels, training, validation, test
    )

    if normalise:
        # fit standard scaler on training data only
        transform.fit_standard_scaler(train_data)
        train_data = transform.transform_standard_scaler(train_data)

        # apply fit to testing data
        test_data = transform.transform_standard_scaler(test_data)

    dmo_data_transform, dmo_label_transform = transforms

    if dmo_data_transform is not None:
        for transform in dmo_data_transform:
            train_data = transform(train_data)
            test_data = transform(test_data)

    if dmo_label_transform is not None:
        for transform in dmo_label_transform:
            train_label = transform(train_label)
            test_label = transform(test_label)

    return ((train_data, train_label), (test_data, test_label))
