from DatasetManager.MSDataLoader import MSDataLoader
from DatasetManager.helper.enum_def import Site, MileStone
from DatasetManager.helper.named_tuple_def import SplitRatio, SplitData, Condition
from Dataset.PatientDataset import FatigueDMODataset
from torch.utils.data import DataLoader

MILESTON = MileStone.T1
SITE = Site.MS10
BATCH_SIZE = 5

dmo_features = [
    "NumberStrides",
    "Duration",
    "AverageCadence",
    "AverageStrideSpeed",
    "AverageStrideLength",
]
ms_dl = MSDataLoader()
ml = ms_dl.ml

condition_value = MILESTON.value.lower()
f_ids = ml.filter_csv_data(
    ml.metadata, Condition("visit.number", "==", condition_value)
)
f_ids_site = ml.filter_by_site(SITE, ml.get_all_ids(f_ids))


training, validation, test = ml.split_data(
    f_ids_site, SplitRatio(training=0.7, validation=0.15, test=0.15)
)

training_patients = ms_dl.instantiate_patients(training, dmo_features, MILESTON)
validation_patients = ms_dl.instantiate_patients(validation, dmo_features, MILESTON)
test_patients = ms_dl.instantiate_patients(test, dmo_features, MILESTON)

training_data = FatigueDMODataset(training_patients, MILESTON, len(dmo_features))

training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

for batch, (X, y) in enumerate(training_dataloader):
    print(f"batch: {batch}\ninput_data: {X}\ntarget_data: {y}")
