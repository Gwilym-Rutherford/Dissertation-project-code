from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    model_type: str

    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    epochs: int

    optimiser: str
    loss_fn: str
    learning_rate: float
    
    notes: str = "No notes"
