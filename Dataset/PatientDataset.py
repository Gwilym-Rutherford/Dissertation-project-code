from torch.utils.data import Dataset
from DatasetManager.Participant import Participant


class FatigueDMODataset(Dataset):
    def __init__(
        self,
        patient_dmo_features,
        patient_id,
        transform=None,
        target_transform=None,
    ):
        self.patient_dmo_features = patient_dmo_features
        self.patient_id = patient_id
        self.transform = transform
        self.target_transform = target_transform

        self.patient = Participant(self.patient_id)
        self.fatigue_results = self.patient.get_csv_column(
            self.patient.filter_csv_data(
                self.patient.get_participant_metadata(), "MFISTO1N", ">", 0
            ),
            "MFISTO1N",
        )


    def __len__(self):
        return len(self.fatigue_results)

    def __getitem__(self, idx):
        questionaire = self.questionaire[idx]
        result = self.fatigue_results[idx]

        if self.transform:
            questionaire = self.transform(questionaire)
        if self.target_transform:
            result = self.target_transform(result)

        return questionaire, result
