from DatasetManager.Participant import Participant

from DatasetManager.Enums.Day import Day
from DatasetManager.Enums.MileStone import MileStone

import orjson
import torch
import os


class Preprocess(object):
    def __init__(self, features):
        self.features = features

    def process_data(self, participant_id):
        part = Participant(participant_id)
        features = self.features

        days = list(Day)
        milestones = [
            milestone for milestone in MileStone if milestone != MileStone.DMO
        ]
        # milestones = [MileStone.T3]

        day_feature_tensor = torch.zeros(len(days) * len(milestones), len(features))

        for m_index, milestone in enumerate(milestones):
            for d_index, day in enumerate(days):
                tensor_index = (m_index * len(days)) + d_index
                feature_tensor = torch.zeros(len(features), dtype=torch.float32)

                data = part.get_day_milestone_participant_info(day, milestone)
                if data is None:
                    continue

                if os.path.isdir(data["sensordmo"]):
                    data_path = os.path.join(
                        data["sensordmo"],
                        f"{milestone.value}_{day.value}_{participant_id}_WBASO_Output.json",
                    )

                    if not os.path.isfile(data_path):
                        continue

                    with open(data_path, "r") as dmo_data:
                        dmo_json = orjson.loads(dmo_data.read())

                    json_data = dmo_json["WBASO_Output"]["TimeMeasure1"]["Recording1"][
                        "SU"
                    ]["LowerBack"]["WB"]

                    for i_wb, wb in enumerate(json_data):
                        for i_feature, feature in enumerate(features):
                            feature_tensor[i_feature] += wb[feature]

                    feature_tensor.div_(len(json_data))

                day_feature_tensor[tensor_index, :] = feature_tensor
        return day_feature_tensor
