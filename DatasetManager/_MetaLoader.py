from collections import namedtuple
from DatasetManager.helper.enum_def import Site, MileStone
from DatasetManager.helper.named_tuple_def import SplitData, SplitRatio
from typing import TypeAlias

import pandas as pd
import os
import operator
import torch


Ids: TypeAlias = torch.Tensor


class _MetadataLoader:
    def __init__(self, dataset_path: str) -> None:
        self.path = os.path.join(
            dataset_path, "V7.2/Main datasets for analysis T1-T5/MS_dataset_v.7.2.csv"
        )

        reader = pd.read_csv(self.path, chunksize=100000, low_memory=False)
        self.metadata = pd.concat(reader, ignore_index=True)

    def filter_csv_data(
        self, data: pd.DataFrame, *conditions: namedtuple
    ) -> pd.DataFrame | None:
        ops = {
            ">": operator.gt,
            "<": operator.lt,
            "==": operator.eq,
            "!=": operator.ne,
        }

        for condition in conditions:
            mask = ops[condition.operation](
                data[condition.column], condition.value
            )
            processed_data = data[mask]
            if processed_data.empty:
                return None
            data = processed_data
        
        return data

    def get_all_ids(self, data: pd.DataFrame | None = None) -> torch.Tensor:
        if data is None:
            data = self.metadata
        data = data["Local.Participant"].to_list()
        return torch.tensor(list(set(data)))

    def filter_by_site(
        self, site: Site, ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if ids is None:
            ids = self.get_all_ids()
        prefixes = ids.div(1000).floor()
        mask = prefixes == int(site.value)
        return ids[mask]

    def get_patient_data(self, id: int) -> pd.DataFrame | None:
        Condition = namedtuple("Condition", "column operation value")
        filter_condition = Condition("Local.Participant", "==", id)
        return self.filter_csv_data(self.metadata, filter_condition)

    def split_data(self, data: Ids, split: SplitRatio) -> SplitData | None:
        if split.sum() != 1:
            return None

        training_split = round(len(data) * split.training)
        validation_split = round(len(data) * split.validation)
        test_split = len(data) - training_split - validation_split

        return SplitData(
            *torch.split(data, [training_split, validation_split, test_split])
        )
