from .base import BaseLoader
from src.core.types import DMOFeatures, ListIds, CSVData
from src.core.enums import MileStone
from src.core.data_transforms import Transform

import pandas as pd
import torch
import os

class DMOLoader(BaseLoader):
    def __init__(self, config_path: str, milestone: MileStone, dmo_features: DMOFeatures = None) -> None:
        super().__init__(config_path)
        self.path = self.config["paths"]["dmo_data_path"]
        self.milestone = milestone

        if dmo_features is not None:
            self.dmo_features = dmo_features.copy()
        else:
            self.dmo_features = None

    def __call__(self, ids: ListIds) -> CSVData:
        return self.get_dmo_data(ids)

    def get_dmo_data(self, ids: ListIds) -> CSVData:
        dir_list = os.listdir(self.path)
        milestone_dir = [x for x in dir_list if x[:2] == self.milestone.value][0]

        milestone_list = os.listdir(os.path.join(self.path, milestone_dir))
        csv_file_name = [x for x in milestone_list if "daily_agg_all" in x][0]

        csv_path = os.path.join(self.path, milestone_dir, csv_file_name)

        reader = pd.read_csv(csv_path, chunksize=100000, low_memory=False)
        dmo_reader = pd.concat(reader, ignore_index=True)
        
        if ids is None:
            print("no ids where given")
            return dmo_reader

        filtered_by_ids = dmo_reader[dmo_reader["participant_id"].isin(ids)]
        
        if self.dmo_features is not None:
            self.dmo_features.append("participant_id")
        else:
            columns_to_remove = ["visit_type", "measurement_date"]
            all_columns = list(filtered_by_ids.columns)
            self.dmo_features = [column for column in all_columns if column not in columns_to_remove]


        filtered_by_features = filtered_by_ids[self.dmo_features]
        
        split_data_by_id = []
        for id_ in ids:
            filtered = filtered_by_features[filtered_by_features["participant_id"] == id_]
            filtered = filtered.drop(columns=["participant_id"])
            
            normalised = Transform.fix_missing_dmo_data(filtered)

            split_data_by_id.append(torch.from_numpy(normalised.to_numpy()))

        return torch.stack(split_data_by_id)




