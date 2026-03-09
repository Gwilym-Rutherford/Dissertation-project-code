from .base import BaseLoader
from src.core.types import CSVData
from src.core.enums import MileStone, Site

import pandas as pd
import scipy.io as scio
import numpy as np
import pyarrow.parquet as pq
import os
import torch
import orjson
import re


class SensorLoader(BaseLoader):
    def __init__(self, config_path, metadata: CSVData, milestone: MileStone):
        super().__init__(config_path)
        self.path = self.config["paths"]["sensor_data"]
        self.metadata = metadata

        self.site_paths = []
        for site in Site:
            self.site_paths.append(os.path.join(self.path, f"MS{site.value}", "files"))

        self.id_path = {}
        for site_path in self.site_paths:
            try:
                ids = os.listdir(site_path)
            except FileNotFoundError:
                continue
            for id_ in ids:
                self.id_path[int(id_)] = os.path.join(site_path, id_)

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

    def __call__(self, ids):
        return self.get_sensor_data(ids)

    def get_sensor_data(self, ids):

        id_paths = {id_: path for id_, path in self.id_path.items() if id_ in ids}
        print(id_paths)
        sensor_lables = []
        sensor_data = []

        fatigue_dict = self.metadata.set_index(["Local.Participant", "visit.number"])[
            "MFISTO1N"
        ].to_dict()

        for id_, path in id_paths.items():
            temp_sensor_data = []

            for milestone in self.milestone:
                milestone_path = os.path.join(path, milestone.value)
                milstone_data = self._get_sensor_data_by_milestone(milestone_path)

                label = fatigue_dict.get((id_, milestone.value.lower()))

                if milstone_data is not None and not np.isnan(label):
                    sensor_lables.append(label)

                    # flip so that it can have different number of timesteps
                    milstone_data = milstone_data.to_numpy()
                    temp_sensor_data.append(milstone_data)

            sensor_data.extend(temp_sensor_data)

        print(torch.tensor(sensor_lables))
        print(torch.from_numpy(np.array(sensor_data)).shape)

        return 1, 2

    def _get_sensor_data_by_milestone(self, path: str) -> torch.Tensor | None:
        if os.path.isdir(path):
            dir_list = os.listdir(path)
            dir_list = [item for item in dir_list if item != "Lab"]

            data_accumulation = []

            if not dir_list:
                return None

            data = pd.DataFrame()

            for day in dir_list:
                day_path = os.path.join(path, day)
                contents = os.listdir(day_path)

                if not contents:
                    continue

                if "data.parquet" not in contents:
                    self.convert_mat_to_parquet(day_path)

                parquet_path = os.path.join(day_path, "data.parquet")
                raw_data = pq.read_table(parquet_path).to_pandas()

                walking_bout_data = self.extract_walking_bouts(raw_data, day, path)

                data_accumulation.append(raw_data)

            data = pd.concat(data_accumulation)
            return data

        return None

    def extract_walking_bouts(
        self, sensor_data: torch.Tensor, day: str, path: str
    ) -> torch.Tensor:
        dmo_path = os.path.join(path, "..", "dmos", day)
        contents = os.listdir(dmo_path)
        json_file = [x for x in contents if re.search(r"WBASO_Output.json", x)][0]
        json_path = os.path.join(dmo_path, json_file)

        with open(json_path, "rb") as json_file:
            json_data = orjson.loads(json_file.read())

        walking_bout_type = "LevelWB"

        walking_bout_data = json_data["WBASO_Output"]["TimeMeasure1"]["Recording1"]["SU"][
            "LowerBack"
        ][walking_bout_type]

        initial_start_time = sensor_data[0, 0]

        for wb in walking_bout_data:
            start = wb["Start"]
            end = wb["End"]

            relative_start = initial_start_time + start
            relative_end = initial_start_time + end

            

    def convert_mat_to_parquet(self, path: str) -> bool:
        parquet_file = os.path.join(path, "data.parquet")
        mat_file = os.path.join(path, "data.mat")

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

            raw_data_df.to_parquet(parquet_file, index=False)

            with open(os.path.join(path, "metadata.json"), "wb") as meta:
                meta.write(orjson.dumps(raw_metadata))

            return True
