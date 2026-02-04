from DatasetManager.DataLoader import DataLoader
from DatasetManager.MetaLoader import MetaLoader
from DatasetManager.Enums.MileStone import MileStone
from DatasetManager.Enums.Day import Day

import numpy as np
import scipy.io as scio
import os
import json


class SensorLoader(DataLoader):
    def __init__(self, participant_id, metaloader, path=None):
        super().__init__()

        self.participant_id = participant_id
        self.file_prefix = f"MS{str(self.participant_id)[0:2]}"

        if metaloader.get_local_participant(self.participant_id) is None:
            quit(1)

        if path is None:
            path = self.curr_file_path
        self.path = os.path.join(
            path,
            "Sensor data",
            self.file_prefix,
            "files",
            str(self.participant_id),
        )

        if not os.path.isdir(self.path):
            self.path_not_exist(self.path)

    def set_milestone(self, milestone):
        if milestone not in MileStone:
            print("Please provide a valid milestone enum")

        self.milestone = milestone.value
        self.data_path = os.path.join(self.path, self.milestone)

        return self.path

    def get_sensor_data_paths(self, day, milestone=None, dmo=False):
        try:
            if milestone is not None:
                self.set_milestone(milestone)

            paths = os.listdir(self.data_path)
            filtered_milestone_path = [
                path for path in paths if path.endswith(day.value)
            ]

            milestone_file = filtered_milestone_path[0]
            milestone_path = os.path.join(self.data_path, milestone_file)

            if dmo:
                self.set_milestone(MileStone.DMO)
                paths = os.listdir(self.data_path)
                filtered_dmo_path = [path for path in paths if path.endswith(day.value)]

                dmo_file = filtered_dmo_path[0]
                dmo_path = os.path.join(self.data_path, dmo_file)

                return milestone_path, dmo_path

            else:
                return milestone_path

        except IndexError:
            print(f"No day {day.value} found")
            quit(1)

    def get_info_for_algo(self, info_for_algo_path):
        mat = scio.loadmat(info_for_algo_path)
        mat_info_base = mat["infoForAlgo"][0][0]["TimeMeasure1"][0][0]

        info = {}
        for attr in mat_info_base.dtype.names:
            value = mat_info_base[attr][0]
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
            elif isinstance(value, np.str_):
                value = str(value)

            info[attr] = value

        return info

    def get_sensor_data(self, data_path):
        mat = scio.loadmat(data_path)
        mat_sensor_base = mat["data"][0][0]["TimeMeasure1"][0][0]["Recording1"][0][0]

        start_date_time = mat_sensor_base["StartDateTime"][0]
        time_zone = mat_sensor_base["TimeZone"][0]

        mat_sensor_base = mat_sensor_base["SU"][0][0]["LowerBack"][0][0]

        sample_frequency = mat_sensor_base["Fs"][0][0]
        time_stamp = mat_sensor_base["Timestamp"]
        acceleration = mat_sensor_base["Acc"]
        gyroscope = mat_sensor_base["Gyr"]

        return {
            "StartDateTime": str(start_date_time),
            "TimeZone": str(time_zone),
            "Fs": sample_frequency,
            "TimeStamp": time_stamp.flatten(),
            "Acc": acceleration,
            "Gyr": gyroscope,
        }

    def get_walking_bout_analysis_dmo(self, day, milestone):
        _, dmo = self.get_sensor_data_paths(day, milestone, dmo=True)

        dmo_path = os.path.join(
            dmo,
            f"{milestone.value}_{day.value}_{self.participant_id}_WBASO_Output.json",
        )

        with open(dmo_path, "r") as json_file:
            dmo_data = json.load(json_file)

        return dmo_data


def main():
    sensorloader = SensorLoader(10376)
    raw_data, dmo_data = sensorloader.get_sensor_data_paths(
        Day.DAY1, milestone=MileStone.T3, dmo=True
    )

    sensorloader.show_dir_contents(dmo_data)
    sensorloader.show_dir_contents(raw_data)


if __name__ == "__main__":
    main()
