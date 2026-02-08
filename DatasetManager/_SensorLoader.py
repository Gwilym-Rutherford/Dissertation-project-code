from operator import itemgetter

import scipy.io as scio
import pandas as pd
import os
import orjson
import torch


class _SensorLoader:
    def __init__(self, dataset_path):
        self.sensor_path = os.path.join(
            dataset_path,
            "Sensor data",
        )

    def get_patient_dmo_data(self, id, features=None):
        file_prefix = self.file_prefix = f"MS{str(id)[0:2]}"
        patient_path = os.path.join(
            self.sensor_path, file_prefix, "files", str(id), "dmos"
        )

        if not os.path.isdir(patient_path):
            return None

        sub_dir = os.listdir(patient_path)

        dmo_data = {}
        for dir_ in sub_dir:
            path = os.path.join(patient_path, dir_, f"{dir_}_{id}_WBASO_Output.json")
            if os.path.isfile(path):
                with open(path, "r") as json:
                    json_data = orjson.loads(json.read())

                if not features:
                    dmo_data[dir_] = json_data
                else:
                    json_data = json_data["WBASO_Output"]["TimeMeasure1"]["Recording1"][
                        "SU"
                    ]["LowerBack"]["WB"]
                    dmo_data[dir_] = self._average_features_per_day(json_data, features)
            else:
                dmo_data[dir_] = None

        return dmo_data

    def get_patient_raw_data(self, id):
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
                path = os.path.join(patient_path, milestone, day, "data.mat")
                raw_data[milestone][day] = self._extract_raw_data(path)

        return raw_data

    def _average_features_per_day(self, data, features):
        df = pd.DataFrame(data)    
        feature_tensor = torch.from_numpy(df[features].values)
        return feature_tensor.mean(dim=0)

    def _extract_raw_data(self, path):
        if not os.path.isfile(path):
            return None

        mat = scio.loadmat(path)
        mat_sensor_base = mat["data"][0][0]["TimeMeasure1"][0][0]["Recording1"][0][0]

        start_date_time = mat_sensor_base["StartDateTime"][0]
        time_zone = mat_sensor_base["TimeZone"][0]

        mat_sensor_base = mat_sensor_base["SU"][0][0]["LowerBack"][0][0]

        sample_frequency = mat_sensor_base["Fs"][0][0]
        time_stamp = torch.tensor(mat_sensor_base["Timestamp"])
        acceleration = torch.tensor(mat_sensor_base["Acc"])
        gyroscope = torch.tensor(mat_sensor_base["Gyr"])

        return {
            "StartDateTime": str(start_date_time),
            "TimeZone": str(time_zone),
            "Fs": sample_frequency,
            "TimeStamp": time_stamp.flatten(),
            "Acc": acceleration,
            "Gyr": gyroscope,
        }
