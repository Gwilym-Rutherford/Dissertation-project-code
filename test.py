from DatasetManager.MSDataLoader import MSDataLoader

dmo_features = [
        "NumberStrides",
        "Duration",
        "AverageCadence",
        "AverageStrideSpeed",
        "AverageStrideLength",
    ]

d = MSDataLoader()

patient = d.get_patient(10376, dmo_features)

print(patient)