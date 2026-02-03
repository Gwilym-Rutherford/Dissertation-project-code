from DataLoader import DataLoader
from MetaLoader import MetaLoader
from SensorLoader import SensorLoader

from Enums.Day import Day
from Enums.MileStone import MileStone

import scipy.io as scio
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


# def main():
pl = ParticipantLoader()
data = pl.get_day_milestone_participant_info(Day.DAY1, MileStone.T3, 10376)
print(data)

raw_data_path = os.path.join(data["sensordata"], "data.mat")
info_data_path = os.path.join(data["sensordata"], "infoForAlgo.mat")

mat = scio.loadmat(raw_data_path)
mat2 = scio.loadmat(info_data_path)


sl = SensorLoader(10376)
print(sl.get_info_for_algo(info_data_path))
print(sl.get_sensor_data(raw_data_path))

# print(mat["data"][0][0]["TimeMeasure1"][0][0]["Recording1"][0][0]["SU"][0][0]["LowerBack"][0][0].dtype)
# print(mat["data"][0][0]["TimeMeasure1"][0][0]["Recording1"][0][0]["SU"][0][0]["LowerBack"][0][0])

# info_file = scio.loadmat(info_data_path)
# print(info_file)
# print(info_file['Weight'])


# if __name__ in "__main__":
#     main()
