from dataclasses import dataclass
from torch.optim import Optimizer

@dataclass
class ModelConfig:
    name: str
    model_type: str

    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    epochs: int
    batch_size: int

    optimiser: Optimizer
    loss_fn: callable
    learning_rate: float
    
    notes: str = "No notes"
