import torch

class Normaliser():
    def __init__(self, scaler):
        self.scaler = scaler
        
        
    def scaler_fit_transform(self, data):
        
        data_to_fit = data
        
        if len(data_to_fit.shape) < 4:
            data_to_fit = data_to_fit.unsqueeze(dim=0)

        patients, visit, day, features = data_to_fit.shape
        data_2d = data_to_fit.reshape(patients * visit * day, features)
        data_2d_scaled = self.scaler.fit_transform(data_2d)
        data_imputed = data_2d_scaled.reshape(patients, visit, day, features)

        data_imputed = torch.squeeze(torch.from_numpy(data_imputed), dim=0)
        return data_imputed
    
    def scaler_transform(self, data):

        data_to_transform = data
        
        if len(data.shape) < 4:
            data_to_transform = data_to_transform.unsqueeze(dim=0)


        patients, visits, days, features = data_to_transform.shape
        data_2d = data_to_transform.reshape(patients * visits * days, features)
        data_2d_scaled = self.scaler.transform(data_2d)
        data_transformed = data_2d_scaled.reshape(patients, visits, days, features)
        
        data_transformed = torch.squeeze(torch.from_numpy(data_transformed), dim=0)
        
        return data_transformed
        
        
    def scaler_inverse_labels(self, data):
        data_to_transform = data.reshape(-1, 1)
        data_inversed = self.scaler.inverse_transform(data_to_transform)
        return torch.tensor(data_inversed).squeeze()
        
        
    