from src.patient_data_dispatcher import PatientDataDispatcher, PatientDataType
from src.core.enums import MileStone
from torchvision.transforms import Compose
from src.core.data_transforms import Transform

import torch

# for debuggin print matrix nicely with no line breaks
torch.set_printoptions(linewidth=200, sci_mode=False, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("getting data")
pdd = PatientDataDispatcher("config/config.yaml", None, MileStone.T3)

ids = list(set(pdd.metadata["Local.Participant"].to_list()))
dmo_data, dmo_labels = pdd.get_patient_data(PatientDataType.SENSOR, ids=ids)

# config = lstm_regression

# dmo_data_transform = Compose([Transform.center_dmo_data])
# dmo_label_transform = Compose([Transform.normalise_dmo_label])

# print("loading into dataloaders")
# transforms = (dmo_data_transform, dmo_label_transform)
# train, validation, test = dmo_into_dataloader(
#     dmo_data, dmo_labels, config.batch_size, transforms, uniform_method=None
# )

# model = DMOLSTM(config).to(device=device)
# optimiser = config.optimiser(model.parameters(), lr=config.learning_rate)

# print("Beginning training")
# lstm_train = LSTMRegressionTrain(
#     model=model, optimiser=optimiser, device=device, config=config
# )

# lstm_train.train(train, validation, test)
