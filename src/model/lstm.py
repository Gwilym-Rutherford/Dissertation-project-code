import torch
import torch.nn as nn
from src.logger import ModelConfig

# Notes
# for LSTMS there are 2 outputs the short term memory and the long term memory
# the final output should be from the short term memory


class DMOLSTM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_size = config.num_layers

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(config.hidden_size, config.output_size)
        self.relu = nn.ReLU()


    def forward(self, x):

        x = x.to(dtype=torch.float32)

        batch_size = x.size(0)
        device = x.device

        h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size, device=device)

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        last_hidden = h_n[-1]

        out = self.linear(last_hidden)
        out = self.relu(out)

        return out
