import torch


def split_data(
    data: torch.Tensor,
    training: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
) -> tuple[torch.Tensor]:
    if sum([training, validation, test]) != 1:
        return None

    training_split = round(len(data) * training)
    validation_split = round(len(data) * validation)
    test_split = len(data) - training_split - validation_split

    return torch.split(data, [training_split, validation_split, test_split])
