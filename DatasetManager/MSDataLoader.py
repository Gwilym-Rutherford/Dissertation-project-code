from DatasetManager.helper.Helper import Helper
from DatasetManager.Patient import Patient
from DatasetManager._SensorLoader import _SensorLoader
from DatasetManager._MetaLoader import _MetadataLoader

import os


class MSDataLoader:
    def __init__(self):
        self.config = Helper.load_config_yaml(
            os.path.join(os.getcwd(), "config/config.yaml")
        )
        self.sl = _SensorLoader(self.config["paths"]["DatasetManager"])
        self.ml = _MetadataLoader(self.config["paths"]["DatasetManager"])

    def get_metadata(self):
        return self.ml.metadata

    def get_patient(self, id, dmo_features=None):
        return Patient(
            id,
            self.ml.get_patient_data(id),
            self.sl.get_patient_dmo_data(id, dmo_features),
            self.sl.get_patient_raw_data(id)
        )
