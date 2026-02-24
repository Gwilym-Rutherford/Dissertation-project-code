from src.core.types import DMOTensor

import torch


def clean_dmo_data(
    dmo_data: DMOTensor, labels: DMOTensor
) -> tuple[DMOTensor, DMOTensor]:
    mask = []
    for i in range(len(dmo_data)):
        if not torch.isnan(labels[i]) and dmo_data[i].sum().item() > 0:
            mask.append(i)

    labels = labels[mask]
    dmo_data = dmo_data[mask]

    return dmo_data, labels
