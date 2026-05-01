from src.model import DMOLSTM
from src.dataset import DMOFatigueDataset
from torch.utils.data import DataLoader
from src.core.data_transforms import Transform
from torchmetrics.regression import R2Score

from src.train import TrainRegression

import torch


class LSTMRegressionXV:
    def __init__(self, dmo_data, dmo_labels, config, device, k=5, seed=1234):
        self.dmo_data = dmo_data
        self.dmo_labels = dmo_labels
        self.k_folds = k
        self.device = device
        self.config = config

        self.shap_values = []
        self.test_inputs = []
        self.baseline_values = []
        self.R2_values = []

        # shuffle
        patients, *other = self.dmo_data.shape
        generator = torch.Generator()
        generator.manual_seed(seed)
        random_permutation = torch.randperm(patients, generator=generator)

        self.dmo_data = self.dmo_data[random_permutation]
        self.dmo_labels = self.dmo_labels[random_permutation]

    def train_cross_validation(self):
        dmo_data_split = torch.tensor_split(self.dmo_data, self.k_folds)
        dmo_label_split = torch.tensor_split(self.dmo_labels, self.k_folds)

        prediction_arr = []
        actual_arr = []

        for split_i in range(self.k_folds):
            
            model = DMOLSTM(self.config).to(device=self.device)
            optimiser = self.config.optimiser(
                model.parameters(), lr=self.config.learning_rate
            )
            
            test_label = dmo_label_split[split_i]
            test_data = dmo_data_split[split_i]

            train_label = torch.cat(
                [dmo_label_split[i] for i in range(0, self.k_folds) if i != split_i]
            )
            train_data = torch.cat(
                [dmo_data_split[i] for i in range(0, self.k_folds) if i != split_i]
            )

            transform = Transform()
            transform.fit_standard_scaler(train_data)
            train_data = transform.transform_standard_scaler(train_data)

            test_data = transform.transform_standard_scaler(test_data)

            # create datasets and dataloaders
            testing_dataset = DMOFatigueDataset(test_label, test_data)

            train_dataset = DMOFatigueDataset(train_label, train_data)

            test_dataloader = DataLoader(
                testing_dataset, batch_size=self.config.batch_size 
            )
            
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.config.batch_size
            )
            
            trainer = TrainRegression(
                model,
                optimiser,
                self.config,
                train_dataloader,
                test_dataloader,
                self.device,
                verbose=True,
            )
            
            trainer.train_loop()
            prediction, actual = trainer.test_loop()
            
            prediction = torch.cat(prediction).squeeze()
            actual = torch.cat(actual).squeeze()
            
            prediction_arr.append(prediction)
            actual_arr.append(actual)
            
            # get R2 score
            r2 = R2Score()
            self.R2_values.append(r2(prediction, actual).item())
        
        return prediction_arr, actual_arr
                        

