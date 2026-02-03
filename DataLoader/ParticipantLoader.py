from DataLoader import DataLoader
from MetaLoader import MetaLoader
from SensorLoader import SensorLoader

from Enums.Day import Day
from Enums.MileStone import MileStone

import os


class ParticipantLoader(DataLoader):
    def __init__(self):
        super().__init__()

    def get_day_milestone_participant_info(self, day, milestone, participant_id):
        ml = MetaLoader()
        sl = SensorLoader(participant_id)

        metadata = ml.get_local_participant(participant_id)
        sensordata, sensordmo = sl.get_milestone_data_path(day, milestone, dmo=True)

        return {"metadata": metadata, "sensordata": sensordata, "sensordmo": sensordmo}

    def get_participant_sensor_data(self, day, milestone, participant_id):
        data = self.get_day_milestone_participant_info(day, milestone, participant_id)

        raw_data_path = os.path.join(data["sensordata"], "data.mat")
        info_data_path = os.path.join(data["sensordata"], "infoForAlgo.mat")

        sl = SensorLoader(participant_id)

        return {
            "infoForAlgo": sl.get_info_for_algo(info_data_path),
            "sensorData": sl.get_sensor_data(raw_data_path),
        }


pl = ParticipantLoader()
data = pl.get_day_milestone_participant_info(Day.DAY1, MileStone.T3, 10376)
print(data)

print(pl.get_participant_sensor_data(Day.DAY1, MileStone.T3, 10376))
