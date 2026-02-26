from .base import BaseLoader
from src.core.types import DMOFeatures, ListIds, CSVData
from src.core.enums import MileStone
from src.core.data_transforms import Transform

import pandas as pd
import torch
import os
import re


class DMOLoader(BaseLoader):
    def __init__(
        self,
        config_path: str,
        milestone: MileStone,
        metadata: CSVData,
        dmo_features: DMOFeatures = None,
    ) -> None:
        super().__init__(config_path)
        self.path = self.config["paths"]["dmo_data_path"]
        self.metadata = metadata

        if milestone == MileStone.ALL:
            self.milestone = [
                MileStone.T1,
                MileStone.T2,
                MileStone.T3,
                MileStone.T4,
                MileStone.T5,
            ]
        else:
            self.milestone = [milestone]

        if dmo_features is not None:
            self.dmo_features = dmo_features.copy()
        else:
            self.dmo_features = None

    def __call__(self, ids: ListIds) -> CSVData:
        return self.get_dmo_data(ids)

    def get_dmo_data(self, ids: ListIds) -> CSVData:

        dir_list = os.listdir(self.path)

        dmo_data = pd.DataFrame()

        # get respected milstone data into one data frame
        for milestone in self.milestone:
            milestone_dir = [x for x in dir_list if x[:2] == milestone.value][0]
            milestone_dir_list = os.listdir(os.path.join(self.path, milestone_dir))
            csv_file = [
                x for x in milestone_dir_list if re.match(r".*daily_agg_all.*.csv$", x)
            ][0]
            csv_path = os.path.join(self.path, milestone_dir, csv_file)

            reader = pd.read_csv(csv_path, chunksize=100000, low_memory=False)
            dmo_reader = pd.concat(reader, ignore_index=True)

            dmo_data = pd.concat([dmo_data, dmo_reader], ignore_index=True)

        if ids is None:
            print("get_dmo_data() needs a list or set of ids")
            quit(-1)

        filtered_by_ids = dmo_data[dmo_data["participant_id"].isin(ids)]

        # get selected dmo features or all
        if self.dmo_features is not None:
            self.dmo_features += ["participant_id", "visit_type"]
        # gets all dmo columns and removes unecessary columns
        else:
            columns_to_remove = ["measurement_date"]
            all_columns = list(filtered_by_ids.columns)
            self.dmo_features = [
                column for column in all_columns if column not in columns_to_remove
            ]

        filtered_dmo_data = filtered_by_ids[self.dmo_features]

        # clean data to fixed size
        dmo_data = []
        dmo_labels = []

        ids = list(set(ids))
        ids.sort()

        # if there is no tensor data or fatigue value then skip
        for id_ in ids:
            rows_with_id = filtered_dmo_data[filtered_dmo_data["participant_id"] == id_]
            rows_with_id = rows_with_id.drop(columns=["participant_id"])

            for milestone in self.milestone:
                row_filtered_by_milestone = rows_with_id[
                    rows_with_id["visit_type"] == milestone.value
                ]

                if row_filtered_by_milestone.empty:
                    continue

                row_fixed_size = Transform.fix_missing_dmo_data(
                    row_filtered_by_milestone
                )
                row_fixed_size = row_fixed_size.drop(columns=["visit_type"])
                dmo_data_tensor = torch.from_numpy(row_fixed_size.to_numpy())

                dmo_label = self.metadata[
                    (self.metadata["Local.Participant"] == id_)
                    & (self.metadata["visit.number"] == milestone.value.lower())
                ]["MFISTO1N"].item()

                if pd.isna(dmo_label):
                    continue
                
                dmo_data.append(dmo_data_tensor)
                dmo_labels.append(dmo_label)

       
        return torch.stack(dmo_data), torch.tensor(dmo_labels)
