from DatasetManager import MSDataLoader, MileStone, Site

# from Dataset.PatientDataset import FatigueDMODataset
# from torch.utils.data import DataLoader


CONFIG_PATH = "config/config.yaml"
MILESTON = MileStone.T3
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
# patient_10376 = ms_dl.get_patient(10376, MileStone.T1, dmo_features, skip_raw=True)
# print(patient_10376)

# patient_10377 = ms_dl.get_patient(10377, MileStone.T1, dmo_features, skip_raw=True)
# print(patient_10377)

ms10_patients = ms_dl.instantiate_patients(
    ms_dl.ml.filter_by_site(Site.MS10, ms_dl.ml.get_all_ids()),
    dmo_features,
    MileStone.T1,
)

print(ms10_patients[10376].sensor_dmo_data)


# ms_dl = MSDataLoader()
# ml = ms_dl.ml

# condition_value = MILESTON.value.lower()
# f_ids = ms_dl.filter_csv_data(
#     ml.metadata, Condition("visit.number", "==", condition_value)
# )
# f_ids_site = ml.filter_by_site(SITE, ml.get_all_ids(f_ids))


# training, validation, test = ml.split_data(
#     f_ids_site, SplitRatio(training=0.7, validation=0.15, test=0.15)
# )

# training_patients = ms_dl.instantiate_patients(training, dmo_features, MILESTON)
# validation_patients = ms_dl.instantiate_patients(validation, dmo_features, MILESTON)
# test_patients = ms_dl.instantiate_patients(test, dmo_features, MILESTON)

# training_data = FatigueDMODataset(training_patients, MILESTON, len(dmo_features))

# training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

# for batch, (X, y) in enumerate(training_dataloader):
#     print(f"batch: {batch}\ninput_data: {X}\ntarget_data: {y}")
