from collections import namedtuple
from .base import BaseLoader


import pandas as pd
import torch



class MetadataLoader(BaseLoader):
    def __init__(self, config_path) -> None:
        super().__init__(config_path)
        self.path = self.config["paths"]["main_dataset_path"]

        reader = pd.read_csv(self.path, chunksize=100000, low_memory=False)
        self.metadata = pd.concat(reader, ignore_index=True)

    def get_all_ids(self, data: pd.DataFrame | None = None) -> torch.Tensor:
        if data is None:
            data = self.metadata
        data = data["Local.Participant"].to_list()
        return torch.tensor(list(set(data)))

    def get_patient_data(self, id: int) -> pd.DataFrame | None:
        Condition = namedtuple("Condition", "column operation value")
        filter_condition = Condition("Local.Participant", "==", id)
        return self.filter_csv_data(self.metadata, filter_condition)

