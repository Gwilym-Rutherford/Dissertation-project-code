from DatasetManager.MetaLoader import MetaLoader
from DatasetManager.SensorLoader import SensorLoader
from DatasetManager.DataLoader import DataLoader

import os


class Participant(DataLoader):
    def __init__(self, participant_id):
        super().__init__()

        self.path = self.config["paths"]["DatasetManager"]

        self.participant_id = participant_id
        self.ml = MetaLoader()
        self.sl = SensorLoader(participant_id, self.ml)

    def get_day_milestone_participant_info(self, day, milestone):

        metadata = self.ml.get_local_participant(self.participant_id)
        sensordata, sensordmo = self.sl.get_sensor_data_paths(day, milestone, dmo=True)

        return {"metadata": metadata, "sensordata": sensordata, "sensordmo": sensordmo}

    def get_participant_sensor_data(self, day, milestone):
        data = self.get_day_milestone_participant_info(day, milestone)

        raw_data_path = os.path.join(data["sensordata"], "data.mat")
        info_data_path = os.path.join(data["sensordata"], "infoForAlgo.mat")

        return {
            "infoForAlgo": self.sl.get_info_for_algo(info_data_path),
            "sensorData": self.sl.get_sensor_data(raw_data_path),
        }

    def get_participant_metadata(self):
        return self.filter_csv_data(
            self.ml.metadata,
            "Local.Participant",
            "==",
            self.participant_id,
        )
