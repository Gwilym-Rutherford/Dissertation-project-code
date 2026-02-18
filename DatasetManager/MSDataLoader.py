from .Patient import Patient
from .SensorLoader import SensorLoader
from .MetaLoader import MetadataLoader
from .DMOLoader import DMOLoader
from .helper.enum_def import MileStone, Site
from .helper.named_tuple_def import SplitData, SplitRatio, Condition
from .types import Ids, Patients, DMOFeatures, CSVData
from torchvision.transforms import Compose
from .helper.transforms import Transform
from Dataset.PatientDataset import FatigueDMODataset
from torch.utils.data import DataLoader


import torch


class MSDataLoader:
    def __init__(self, config_path) -> None:
        self.ml = MetadataLoader(config_path)
        self.sl = SensorLoader(config_path)
        self.dl = DMOLoader(config_path)

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
            self.dl.get_patient_dmo_data(id, dmo_features, milestone),
            self.sl.get_patient_raw_data(id, skip=skip_raw, write_parquet=False),
        )

    def instantiate_patients(
        self, ids: Ids, dmo_features: DMOFeatures, milestone: MileStone
    ) -> Patients:
        patients = []
        for id in ids:
            patients.append(
                self.get_patient(
                    int(id), milestone, dmo_features=dmo_features, skip_raw=True
                )
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

    def train_validation_test(
        self,
        ids: Ids,
        split: SplitRatio,
        dmo_features: DMOFeatures,
        milestone: MileStone,
    ) -> tuple[Patients, Patients, Patients]:

        training, validation, test = self.split_data(ids, split)

        training_patients = self.instantiate_patients(training, dmo_features, milestone)
        validation_patients = self.instantiate_patients(
            validation, dmo_features, milestone
        )
        test_patients = self.instantiate_patients(test, dmo_features, milestone)

        return training_patients, validation_patients, test_patients

    def get_patient_dmo_dataloaders(
        self,
        milestone: MileStone,
        site: Site,
        dmo_features: DMOFeatures,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        condition_value = milestone.value.lower()
        f_ids = self.ml.filter_csv_data(
            self.ml.metadata, Condition("visit.number", "==", condition_value)
        )
        f_ids_site = self.ml.filter_by_site(site, self.ml.get_all_ids(f_ids))
        # f_ids_site = ml.get_all_ids()

        training_patients, validation_patients, test_patients = (
            self.train_validation_test(
                f_ids_site,
                SplitRatio(training=0.7, validation=0.15, test=0.15),
                dmo_features,
                milestone,
            )
        )

        # setup dataloaders and datasets
        transform_compose = Compose([Transform.dmo_normalise, Transform.mask_data])

        training_data = FatigueDMODataset(
            training_patients,
            milestone,
            len(dmo_features),
            transform=transform_compose,
            target_transform=Transform.dmo_label_normalise,
        )

        validation_data = FatigueDMODataset(
            validation_patients,
            milestone,
            len(dmo_features),
            transform=transform_compose,
            target_transform=Transform.dmo_label_normalise,
        )

        test_data = FatigueDMODataset(
            test_patients,
            milestone,
            len(dmo_features),
            transform=transform_compose,
            target_transform=Transform.dmo_label_normalise,
        )

        training_dataloader = DataLoader(training_data, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        return training_dataloader, validation_dataloader, test_dataloader
