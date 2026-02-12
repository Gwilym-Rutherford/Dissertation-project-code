from DatasetManager.helper.Helper import Helper
from DatasetManager._Patient import _Patient
from DatasetManager._SensorLoader import _SensorLoader
from DatasetManager._MetaLoader import _MetadataLoader
from DatasetManager.helper.enum_def import MileStone

import pandas as pd
import os
import torch


class MSDataLoader:
    def __init__(self) -> None:
        self.config = Helper.load_config_yaml(
            os.path.join(os.getcwd(), "config/config.yaml")
        )
        self.sl = _SensorLoader(self.config["paths"]["DatasetManager"])
        self.ml = _MetadataLoader(self.config["paths"]["DatasetManager"])

    def get_metadata(self) -> pd.DataFrame:
        return self.ml.metadata

    def get_patient(
        self,
        id: int,
        milestone: MileStone,
        dmo_features: list[str] | None = None,
        skip_raw: bool = False,
        write_parquet=False,
    ) -> _Patient:
        return _Patient(
            id,
            self.ml.get_patient_data(id),
            self.sl.get_patient_dmo_data(id, dmo_features, milestone),
            self.sl.get_patient_raw_data(id, skip=skip_raw, write_parquet=False),
        )

    def instantiate_patients(
        self, ids: torch.Tensor, dmo_features: list[str], milestone: MileStone
    ) -> list[_Patient]:
        patients = []
        for id in ids:
            patients.append(
                self.get_patient(int(id), milestone, dmo_features=dmo_features, skip_raw=True)
            )
        return patients
