import torch
import torch.nn as nn

# Notes
# for LSTMS there are 2 outputs the short term memory and the long term memory
# the final output should be from the short term memory


class DMOLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_size = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, x):

        batch_size = x.size(0)
        device = x.device

        h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size, device=device)

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        last_hidden = h_n[-1]

        out = self.linear(last_hidden)

        return out
