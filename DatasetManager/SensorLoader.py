from .helper.enum_def import MileStone
from .helper.named_tuple_def import Condition
from .base import BaseLoader

import scipy.io as scio
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import orjson
import torch
import gc


class SensorLoader(BaseLoader):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)

    def get_patient_raw_data(
        self, id: int, skip: bool = False, write_parquet=False
    ) -> dict | None:
        if skip:
            return None

        file_prefix = self.file_prefix = f"MS{str(id)[0:2]}"
        patient_path = os.path.join(self.sensor_path, file_prefix, "files", str(id))

        milestone_list = os.listdir(patient_path)
        milestones = [sub_dir for sub_dir in milestone_list if sub_dir != "dmos"]

        raw_data = {}
        for milestone in milestones:
            raw_data[milestone] = {}
            day_list = os.listdir(os.path.join(patient_path, milestone))
            for day in day_list:
                if day == "Lab":
                    continue
                path = os.path.join(patient_path, milestone, day)
                raw_data[milestone][day] = self._extract_raw_data(
                    path, write_parquet=write_parquet
                )
                gc.collect()

        return raw_data

    def _average_features_per_day(
        self, data: dict, features: list[str]
    ) -> torch.Tensor:
        df = pd.DataFrame(data)
        feature_tensor = torch.from_numpy(df[features].values)
        return feature_tensor.mean(dim=0)

    def _extract_raw_data(self, path: str, write_parquet=False) -> pd.DataFrame | None:
        parquet_file = os.path.join(path, "data.parquet")
        mat_file = os.path.join(path, "data.mat")

        if os.path.isfile(parquet_file):
            return pq.read_table(parquet_file).to_pandas()

        if os.path.isfile(mat_file):
            mat = scio.loadmat(mat_file)
            mat_sensor_base = mat["data"][0][0]["TimeMeasure1"][0][0]["Recording1"][0][
                0
            ]

            start_date_time = mat_sensor_base["StartDateTime"][0]
            time_zone = mat_sensor_base["TimeZone"][0]

            mat_sensor_base = mat_sensor_base["SU"][0][0]["LowerBack"][0][0]

            sample_frequency = mat_sensor_base["Fs"][0][0]
            time_stamp = np.array(mat_sensor_base["Timestamp"]).flatten()
            acceleration = np.array(mat_sensor_base["Acc"])
            gyroscope = np.array(mat_sensor_base["Gyr"])

            raw_metadata = {
                "start_date_time": str(start_date_time),
                "time_zone": str(time_zone),
                "fs": str(sample_frequency),
            }

            raw_data = {
                "time_stamp": time_stamp,
                "acc_x": acceleration[:, 0],
                "acc_y": acceleration[:, 1],
                "acc_z": acceleration[:, 2],
                "gyr_x": gyroscope[:, 0],
                "gyr_y": gyroscope[:, 1],
                "gyr_z": gyroscope[:, 2],
            }

            raw_data_df = pd.DataFrame(raw_data)

            if write_parquet:
                raw_data_df.to_parquet(parquet_file, index=False)

                with open(os.path.join(path, "metadata.json"), "wb") as meta:
                    meta.write(orjson.dumps(raw_metadata))

            return raw_data_df
