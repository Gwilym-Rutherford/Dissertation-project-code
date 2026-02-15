from DatasetManager import MSDataLoader, MileStone, Site, SplitRatio, Condition
from DatasetManager.helper.transforms import Transform
from Dataset.PatientDataset import FatigueDMODataset

import math

from torch.utils.data import DataLoader


CONFIG_PATH = "config/config.yaml"
MILESTON = MileStone.T2
SITE = Site.MS10
BATCH_SIZE = 1

dmo_features = [
    "cadence_all_avg_d",
    "wbdur_all_avg_d",
    "strdur_30_avg_d",
    "cadence_30_avg_d",
    "ws_30_avg_d",
]

ms_dl = MSDataLoader(CONFIG_PATH)
ml = ms_dl.ml

condition_value = MILESTON.value.lower()
f_ids = ms_dl.ml.filter_csv_data(
    ml.metadata, Condition("visit.number", "==", condition_value)
)
f_ids_site = ml.filter_by_site(SITE, ml.get_all_ids(f_ids))


training, validation, test = ms_dl.split_data(
    f_ids_site, SplitRatio(training=0.7, validation=0.15, test=0.15)
)

training_patients = ms_dl.instantiate_patients(training, dmo_features, MILESTON)
validation_patients = ms_dl.instantiate_patients(validation, dmo_features, MILESTON)
test_patients = ms_dl.instantiate_patients(test, dmo_features, MILESTON)

training_data = FatigueDMODataset(
    training_patients, MILESTON, len(dmo_features), transform=Transform.dmo_scale
)

training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

for batch, (X, y) in enumerate(training_dataloader):
    if math.isnan(y.item()) or X.shape == (1, 1):
        continue
    print(f"batch: {batch}\ninput_data: {X}\ntarget_data: {y.item()}")
