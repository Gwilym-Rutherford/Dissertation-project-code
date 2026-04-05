import numpy as np
import torch


class TrainRegression:
    def __init__(
        self,
        model,
        optimiser,
        config,
        training_dataloader,
        testing_dataloader,
        device,
        verbose=False,
    ):
        self.model = model
        self.optimiser = optimiser
        self.config = config
        self.device = device
        self.training_dataloader = training_dataloader
        self.testing_dataloader = testing_dataloader
        self.verbose = verbose

    def train_loop(
        self,
        label_transform: callable = None,
        data_transform: callable = None,
        pred_transform: callable = None,
    ):
        for epoch in range(self.config.epochs):
            training_loss = []
            self.model.train()
            for data, label in self.training_dataloader:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)

                if label_transform:
                    label = label_transform(label)

                if data_transform:
                    data = data_transform(data)

                self.optimiser.zero_grad()
                pred = self.model(data).squeeze(dim=0)

                if pred_transform:
                    pred = pred_transform(pred)
                    
                loss = self.config.loss_fn(pred, label)
                training_loss.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimiser.step()

            # if self.verbose:
            #     print(f"Epoch {epoch} - Training loss: {np.average(training_loss)}")

    def test_loop(
        self,
        label_transform: callable = None,
        data_transform: callable = None,
        pred_transform: callable = None,
    ):
        model_prediction = []
        actual_value = []

        testing_loss = []
        self.model.eval()
        with torch.no_grad():
            for data, label in self.testing_dataloader:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32).squeeze(dim=0)

                if label_transform:
                    label = label_transform(label)

                if data_transform:
                    data = data_transform(data)

                pred = self.model(data).squeeze(dim=0)

                if pred_transform:
                    pred = pred_transform(pred)

                loss = self.config.loss_fn(pred, label)
                testing_loss.append(loss.item())

                pred = pred.cpu()
                label = label.cpu()

                model_prediction.append(pred[0])
                actual_value.append(label[0])

        if self.verbose:
            print(f"Testing loss: {np.average(testing_loss)}")

        return model_prediction, actual_value
