import torch
import torch.nn as nn
from src.model.model_config_class import ModelConfig

import math


class DMOTransformer(nn.Module):
    def __init__(
        self,
        input_features: int,
        d_model: int,
        nheads: int,
        num_layers: int,
        dim_feedforward: int,
        output_size: int,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.linear_in = nn.Linear(input_features, d_model)
        
        self.positional_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        self.linear_out = nn.Linear(d_model, output_size)

        max_len = 5000
        mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        #seq_len = x.size(1)
        
        x = self.linear_in(x)
        x = self.positional_encoder(x)
        
        #mask = self.causal_mask[:seq_len, :seq_len]
        
        output = self.transformer_encoder(x)
        
        out = self.linear_out(output)
        
        return out
        


# PositionEncoder needed to prevent bag of words for transformer (pytorch's example)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
