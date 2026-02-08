from collections import namedtuple
from DatasetManager.helper.enum_def import Site

import pandas as pd
import os
import operator
import torch


class _MetadataLoader:
    def __init__(self, dataset_path: str) -> None:
        self.path = os.path.join(
            dataset_path, "V7.2/Main datasets for analysis T1-T5/MS_dataset_v.7.2.csv"
        )

        reader = pd.read_csv(self.path, chunksize=100000, low_memory=False)
        self.metadata = pd.concat(reader, ignore_index=True)

    def filter_csv_data(
        self, data: pd.DataFrame, filter_condition: namedtuple
    ) -> pd.DataFrame | None:
        ops = {
            ">": operator.gt,
            "<": operator.lt,
            "==": operator.eq,
            "!=": operator.ne,
        }

        mask = ops[filter_condition.operation](
            data[filter_condition.column], filter_condition.value
        )

        processed_data = data[mask]
        if processed_data.empty:
            return None
        return processed_data

    def get_all_ids(self) -> torch.Tensor:
        ids = self.metadata["Local.Participant"].to_list()
        return torch.tensor(list(set(ids)))

    def get_all_ids_by_site(self, site: Site) -> torch.Tensor:
        ids = self.get_all_ids()
        prefixes = ids.div(1000).floor()
        mask = prefixes == int(site.value)
        return ids[mask]

    def get_patient_data(self, id: int) -> pd.DataFrame | None:
        Condition = namedtuple("Condition", "column operation value")
        filter_condition = Condition("Local.Participant", "==", id)
        return self.filter_csv_data(self.metadata, filter_condition)
