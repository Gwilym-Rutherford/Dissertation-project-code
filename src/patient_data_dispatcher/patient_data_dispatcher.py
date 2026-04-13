from .loaders import MetaLoader, DMOLoader, SensorLoader, MileStoneLoader

from src.core.types import ListIds, DMOFeatures
from src.core.enums import MileStone, PatientDataType, DataFrequency


class PatientDataDispatcher:
    def __init__(
        self,
        config_path: str,
        dmo_features: DMOFeatures = None,
        milestone: MileStone = None,
        data_frequency: DataFrequency = DataFrequency.DAILY,
        filtered: bool = False,
        static_features: list[str] = None,
        physical_subset: bool = True
    ):

        self.metadata = MetaLoader(config_path, MileStone.ALL)(ids=None)

        self.fetcher = {
            PatientDataType.META: MetaLoader(config_path, milestone),
            PatientDataType.DMO: DMOLoader(
                config_path,
                milestone,
                self.metadata,
                data_frequency,
                filtered,
                dmo_features,
            ),
            # PatientDataType.SENSOR: SensorLoader(config_path, self.metadata, milestone),
            PatientDataType.MILESTONE: MileStoneLoader(
                config_path,
                milestone,
                self.metadata,
                data_frequency,
                filtered,
                dmo_features,
                static_features=static_features,
                physical_subset=physical_subset
            ),
        }

    def get_patient_data(self, data_type: PatientDataType, ids: ListIds = None):
        data_loader = self.fetcher.get(data_type)
        return data_loader(ids)
