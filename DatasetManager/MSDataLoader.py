from DatasetManager.helper.Helper import Helper
from DatasetManager._Patient import _Patient
from DatasetManager._SensorLoader import _SensorLoader
from DatasetManager._MetaLoader import _MetadataLoader

import os
import pandas as pd


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
        dmo_features: list[str] | None = None,
        skip_raw: bool = False,
        write_parquet=False,
    ) -> _Patient:
        return _Patient(
            id,
            self.ml.get_patient_data(id),
            self.sl.get_patient_dmo_data(id, dmo_features),
            self.sl.get_patient_raw_data(id, skip=skip_raw, write_parquet=False),
        )
