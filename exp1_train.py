from DatasetManager import MSDataLoader, MileStone, Site, SplitRatio, Condition
from DatasetManager.helper.transforms import Transform
from Dataset.PatientDataset import FatigueDMODataset
from torch.optim import Adam
from torch.nn import HuberLoss
from torchvision.transforms import Compose
from exp1_model import Exp1Model

import torch


from torch.utils.data import DataLoader

# setup constants
dmo_features = [
    "cadence_all_avg_d",
    "wbdur_all_avg_d",
    "strdur_30_avg_d",
    "cadence_30_avg_d",
    "ws_30_avg_d",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG_PATH = "config/config.yaml"
MILESTON = MileStone.T2
SITE = Site.MS10

INPUT_SIZE = len(dmo_features) * 2
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_DIM = 1

LEARNING_RATE = 1e-4

EPOCHS = 100

# get all ms data
ms_dl = MSDataLoader(CONFIG_PATH)
ml = ms_dl.ml

condition_value = MILESTON.value.lower()
f_ids = ms_dl.ml.filter_csv_data(
    ml.metadata, Condition("visit.number", "==", condition_value)
)
f_ids_site = ml.filter_by_site(SITE, ml.get_all_ids(f_ids))

training_patients, validation_patients, test_patients = ms_dl.train_validation_test(
    f_ids_site,
    SplitRatio(training=0.7, validation=0.15, test=0.15),
    dmo_features,
    MILESTON,
)

# setup dataloaders and datasets
transform_compose = Compose([Transform.dmo_normalise, Transform.mask_data])

training_data = FatigueDMODataset(
    training_patients, MILESTON, len(dmo_features), transform=transform_compose
)

validation_data = FatigueDMODataset(
    validation_patients, MILESTON, len(dmo_features), transform=transform_compose
)

test_data = FatigueDMODataset(
    test_patients, MILESTON, len(dmo_features), transform=transform_compose
)

training_dataloader = DataLoader(training_data, batch_size=BATCH_DIM)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_DIM)
test_dataloader = DataLoader(test_data, batch_size=BATCH_DIM)

print("Setup complete, training starting now")

# instantiate model, optimiser and loss function
model = Exp1Model(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device=device)
optimiser = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = HuberLoss()

# train loop
for epoch in range(EPOCHS):
    model.train()
    for index, (input_data, label) in enumerate(training_dataloader):
        # ignore patients that have no available dmo data
        if input_data.shape == (1, 1):
            continue


        input_data = input_data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        output = model(input_data)
        loss = loss_fn(output, label)
        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for index, (input_data, label) in enumerate(validation_dataloader):
            if input_data.shape == (1, 1):
                continue

            input_data = input_data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output = model(input_data)
            loss = loss_fn(output, label)

            val_loss += loss.item()

    print(f"Epoch {epoch} | Train loss: {loss.item()} | Val loss: {val_loss}")

