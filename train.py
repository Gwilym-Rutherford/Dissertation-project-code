from DatasetManager.MSDataLoader import MSDataLoader
from DatasetManager.helper.enum_def import Site

dmo_features = [
    "NumberStrides",
    "Duration",
    "AverageCadence",
    "AverageStrideSpeed",
    "AverageStrideLength",
]

d = MSDataLoader()

patient = d.get_patient(10376, dmo_features=dmo_features, skip_raw=False)



#print(d.ml.get_all_ids_by_site(Site.MS10))

#print(patient.meta_data)
#print(patient.sensor_dmo_data)
#print(patient.sensor_raw_data["T3"])

ml = d.ml

# ml.filter_csv_data(ml.metadata)

# print(patient)
