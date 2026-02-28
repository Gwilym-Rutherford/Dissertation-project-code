from .base import BaseLoader
from src.core.types import CSVData, ListIds
from src.core.enums import MileStone
import pandas as pd

class MetaLoader(BaseLoader):
    def __init__(self, config_path: str, milestone: MileStone) -> None:
        super().__init__(config_path)
        self.path = self.config["paths"]["main_dataset_path"]
        self.milestone = milestone

    def __call__(self, ids: ListIds) -> CSVData:
        return self.get_metadata(ids)

    def get_metadata(self, ids: ListIds) -> CSVData:
        reader = pd.read_csv(self.path, chunksize=100000, low_memory=False)
        metadata = pd.concat(reader, ignore_index=True)

        if self.milestone != MileStone.ALL:
            metadata = metadata[metadata["visit.number"] == self.milestone.value.lower()]

        if ids is None:
            return metadata

        metadata = metadata.set_index("Local.Participant")
            
        return metadata.loc[ids].reset_index()