from src.core.normaliser import Normaliser
from src.model import DMORandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

import torch
import shap

class XVRandomForest():
    def __init__(self, dmo_data, dmo_labels, k = 5, seed=1234,):
        self.dmo_data = dmo_data
        self.dmo_labels = dmo_labels
        self.k_folds = k
        self.rf_model = DMORandomForestRegressor(n_trees=500)
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
        
    def train_cross_validation(self, label_normaliser):
        
        dmo_data_split = torch.tensor_split(self.dmo_data, self.k_folds)
        dmo_label_split = torch.tensor_split(self.dmo_labels, self.k_folds)
        
        prediction_arr = []
        actual_arr = []

        for split_i in range(self.k_folds):
            test_label = dmo_label_split[split_i]
            test_data = dmo_data_split[split_i]

            train_label = torch.cat(
                [dmo_label_split[i] for i in range(0, self.k_folds) if i != split_i]
            )
            train_data = torch.cat(
                [dmo_data_split[i] for i in range(0, self.k_folds) if i != split_i]
            )

            # normalise data, fitting only on training data
            standard_scaler = Normaliser(StandardScaler())
            train_data = standard_scaler.scaler_fit_transform(train_data)
            test_data = standard_scaler.scaler_transform(test_data)
            
            # format data
            patient, visit, features = train_data.shape
            train_input = train_data.reshape(patient * visit, features)
            train_label = train_label.reshape(patient * visit)

            patient, visit, features = test_data.shape
            test_input = test_data.reshape(patient * visit, features)
            test_label = test_label.reshape(patient * visit)

            self.rf_model.train(train_input, train_label)
            score = self.rf_model.score(test_input, test_label)
            print(score)
            
            prediction = torch.tensor(self.rf_model.predict(test_input))
            prediction = label_normaliser.scaler_inverse_labels(prediction)

            actual = label_normaliser.scaler_inverse_labels(test_label)

            prediction_arr += prediction
            actual_arr += actual
            
            # add R2 values for each fold
            r2 = R2Score()
            self.R2_values.append(r2(prediction, actual).item())
            
            # add shap values for each fold
            sklearn_rf_model = self.rf_model.random_forest
            
            test_input_numpy = torch.Tensor.numpy(test_input)
            explainer = shap.TreeExplainer(sklearn_rf_model)
            shap_values = explainer.shap_values(test_input_numpy)
            self.shap_values.append(shap_values)
            self.test_inputs.append(test_input)
            self.baseline_values.append(explainer.expected_value)
            
        prediction = torch.stack(prediction_arr)
        actual = torch.stack(actual_arr)    
        
        return prediction, actual
            