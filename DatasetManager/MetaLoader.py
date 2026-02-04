from DatasetManager.DataLoader import DataLoader

import os


class MetaLoader(DataLoader):
    def __init__(self, path=None):
        super().__init__()

        if path is None:
            path = self.curr_file_path

        self.path = os.path.join(
            path,
            "V7.2/Main datasets for analysis T1-T5/MS_dataset_v.7.2.csv",
        )

        if not os.path.isfile(self.path):
            self.path_not_exist(self.path)

        self.metadata = super().create_csv_data(self.path)

    def get_all_participant_ids(self):
        ids = self.get_csv_column(self.metadata, "Local.Participant")
        return list(set(ids))

    def get_local_participant(self, participant_id):
        participant_data = self.filter_csv_data(
            self.metadata, "Local.Participant", "==", participant_id
        )

        return self.check_empty(participant_data, "particpant")

    def get_n_milestone_metadata(self, participant_id, milestone):
        milestone_data = self.filter_csv_data(
            self.get_local_participant(25238),
            "visit.number",
            "==",
            milestone.value.lower(),
        )

        return self.check_empty(milestone_data, "milestone")
