from .patient_data_types import PatientDataType
from .loaders import MetaLoader, DMOLoader, SensorLoader

from src.core.types import ListIds, DMOFeatures
from src.core.enums import MileStone


class PatientDataDispatcher:
    def __init__(
        self,
        config_path: str,
        dmo_features: DMOFeatures = None,
        milestone: MileStone = None,
    ):

        self.metadata = MetaLoader(config_path, MileStone.ALL)(ids=None)

        self.fetcher = {
            PatientDataType.META: MetaLoader(config_path, milestone),
            PatientDataType.DMO: DMOLoader(
                config_path, milestone, self.metadata, dmo_features
            ),
            PatientDataType.SENSOR: SensorLoader(config_path, self.metadata),
        }

    def get_patient_data(
        self, data_type: PatientDataType, ids: ListIds = None
    ):
        data_loader = self.fetcher.get(data_type)
        return data_loader(ids)
