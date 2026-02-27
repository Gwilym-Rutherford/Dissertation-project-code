from src.logger import ModelConfig
from torch.nn import HuberLoss
from torch.optim import Adam


lstm_regression = ModelConfig(
    name="lstm_regression",
    model_type="LSTM",
    input_size=26,
    hidden_size=128,
    num_layers=2,
    output_size=1,
    batch_size=16,
    epochs=100,
    optimiser=Adam,
    loss_fn=HuberLoss(),
    learning_rate=5e-4,
)
