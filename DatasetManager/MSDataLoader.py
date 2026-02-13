from .Patient import Patient
from .SensorLoader import SensorLoader
from .MetaLoader import MetadataLoader
from .helper.enum_def import MileStone
from .helper.named_tuple_def import SplitData, SplitRatio
from .types import Ids, Patients, DMOFeatures, CSVData

import torch


class MSDataLoader:
    def __init__(self, config_path) -> None:
        self.ml = MetadataLoader(config_path)
        self.sl = SensorLoader(config_path)

    def get_metadata(self) -> CSVData:
        return self.ml.metadata

    def get_patient(
        self,
        id: int,
        milestone: MileStone,
        dmo_features: DMOFeatures = None,
        skip_raw: bool = False,
        write_parquet: bool = False,
    ) -> Patient:
        return Patient(
            id,
            self.ml.get_patient_data(id),
            self.sl.get_patient_dmo_data(id, dmo_features, milestone),
            self.sl.get_patient_raw_data(id, skip=skip_raw, write_parquet=False),
        )

    def instantiate_patients(
        self, ids: Ids, dmo_features: DMOFeatures, milestone: MileStone
    ) -> Patients:
        patients = {}
        for id in ids:
            patients[id.item()] = self.get_patient(
                int(id), milestone, dmo_features=dmo_features, skip_raw=True
            )

        return patients

    def split_data(self, data: Ids, split: SplitRatio) -> SplitData | None:
        if split.sum() != 1:
            return None

        training_split = round(len(data) * split.training)
        validation_split = round(len(data) * split.validation)
        test_split = len(data) - training_split - validation_split

        return SplitData(
            *torch.split(data, [training_split, validation_split, test_split])
        )
