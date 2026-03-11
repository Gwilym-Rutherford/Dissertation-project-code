from src.core.data_transforms import Transform
from src.core.types import DMOTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from ..split_data import split_data



def dmo_for_random_forest(
    dmo_data: DMOTensor,
    dmo_labels: DMOTensor,
    transforms: tuple[list[callable], list[callable]],
    training: float = 0.8,
    # validation set not needed
    validation: float = 0,
    test: float = 0.20,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    dmo_data, dmo_labels = Transform.clean_dmo_data(dmo_data, dmo_labels)

    train_data, validation_data, test_data = split_data(
        dmo_data, training, validation, test
    )

    train_label, validation_label, test_label = split_data(
        dmo_labels, training, validation, test
    )

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
