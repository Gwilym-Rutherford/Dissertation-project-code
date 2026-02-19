from .helper.enum_def import MileStone, Day
from .helper.named_tuple_def import Condition
from .helper.constants import MASK_VALUE
from .base import BaseLoader
from .types import DMOFeatures, DMOTensor

import pandas as pd
import numpy as np
import os
import torch


class DMOLoader(BaseLoader):
    def __init__(
        self, config_path: str, dmo_features: DMOFeatures, milestone: MileStone
    ) -> None:
        super().__init__(config_path)
        self.dmo_path = self.config["paths"]["dmo_data_path"]

    def __call__(self, ids):
        return self.get_patient_dmo_data(ids)

    def get_patient_dmo_data(self, id: int) -> DMOTensor:

        if (self.features or self.milestone) is None:
            return None

        dir_list = os.listdir(self.dmo_path)
        milestone_dir = [x for x in dir_list if x[:2] == self.milestone.value][0]

        milestone_list = os.listdir(os.path.join(self.dmo_path, milestone_dir))
        csv_file_name = [x for x in milestone_list if "daily_agg_all" in x][0]

        csv_path = os.path.join(self.dmo_path, milestone_dir, csv_file_name)

        reader = pd.read_csv(csv_path, chunksize=100000, low_memory=False)
        dmo_reader = pd.concat(reader, ignore_index=True)

        filtered_data = self.filter_csv_data(
            dmo_reader, Condition("participant_id", "==", id)
        )

        if filtered_data is None:
            return torch.tensor([-1])

        dmo_dataframe = self.filter_csv_column(filtered_data, self.features, keep_id=False)
        dmo_dataframe.fillna(MASK_VALUE, inplace=True)

        n_rows = len(Day) - dmo_dataframe.shape[0]
        if n_rows > 0:
            padding = pd.DataFrame(
                MASK_VALUE, index=np.arange(n_rows), columns=dmo_dataframe.columns
            )
            dmo_dataframe = pd.concat([padding, dmo_dataframe], axis=0)

        dmo_tensor = torch.from_numpy(dmo_dataframe.to_numpy())

        if dmo_tensor is None:
            return torch.tensor([-1])

        return dmo_tensor
