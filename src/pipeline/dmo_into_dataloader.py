from src.core.data_transforms import Transform
from src.core.types import DMOTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from src.dataset import DMOFatigueDataset
from .split_data import split_data


def dmo_into_dataloader(
    dmo_data: DMOTensor,
    dmo_labels: DMOTensor,
    batch_size: int,
    training: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
):

    train_data, validation_data, test_data = split_data(
        dmo_data, training, validation, test
    )

    train_label, validation_label, test_label = split_data(
        dmo_labels, training, validation, test
    )

    dmo_data_transform = Compose(
        [Transform.normalise_dmo_data, Transform.mask_dmo_data]
    )
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

    dataloader_training = DataLoader(dataset_training, batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size)
    dataloader_testing = DataLoader(dataset_testing, batch_size=batch_size)

    return (dataloader_training, dataloader_testing, dataloader_validation)
