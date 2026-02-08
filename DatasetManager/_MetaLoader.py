from collections import namedtuple

import os
import pandas as pd
import operator


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

    def get_all_ids(self) -> list[int]:
        ids = self.metadata["Local.Participant"].to_list()
        return list(set(ids))

    def get_patient_data(self, id: int) -> pd.DataFrame | None:
        Condition = namedtuple("Condition", "column operation value")
        filter_condition = Condition("Local.Participant", "==", id)
        return self.filter_csv_data(self.metadata, filter_condition)
