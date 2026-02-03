from DataLoader import DataLoader
from MetaLoader import MetaLoader
from Enums.MileStone import MileStone
from Enums.Day import Day

import numpy as np
import scipy.io as scio
import os

DEFAULT_DATASET_PATH = os.path.join(os.getcwd(), "Sensor data")


class SensorLoader(DataLoader):
    def __init__(self, participant_id, path=None):
        super().__init__()

        self.participant_id = participant_id
        metaloader = MetaLoader()
        if metaloader.get_local_participant(self.participant_id) is None:
            quit(1)

        self.file_prefix = f"MS{str(self.participant_id)[0:2]}"
        self.base_data_path = os.path.join(
            DEFAULT_DATASET_PATH, self.file_prefix, "files", str(self.participant_id)
        )

        if not path:
            self.path = DEFAULT_DATASET_PATH
        else:
            self.path = path

        if not os.path.isdir(self.path):
            self.path_not_exist(self.path)

    def set_milestone(self, milestone):
        if milestone not in MileStone:
            print("Please provide a valid milestone enum")

        self.milestone = milestone.value
        self.data_path = os.path.join(self.base_data_path, self.milestone)

        return self.path

    def get_milestone_data_path(self, day, milestone=None, dmo=False):
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
            "StartDateTime": start_date_time,
            "TimeZone": time_zone,
            "Fs": sample_frequency,
            "TimeStamp": time_stamp,
            "Acc": acceleration,
            "Gyr": gyroscope
        }
        

def main():
    sensorloader = SensorLoader(10376)
    raw_data, dmo_data = sensorloader.get_milestone_data_path(
        Day.DAY1, milestone=MileStone.T3, dmo=True
    )

    sensorloader.show_dir_contents(dmo_data)
    sensorloader.show_dir_contents(raw_data)


if __name__ == "__main__":
    main()
