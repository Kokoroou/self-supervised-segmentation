# TODO: Define the loss function
# TODO: Define the optimizer
# TODO: Define the training loop
# TODO: Define the validation loop
# TODO: Define the main function
from torch import nn

from model import MaskedAutoencoderViT
from config import model_config, data_config
from data import prepare_data

model_arch = "default"
process_type = "default"


def main():
    # Load the model architecture
    config = model_config["default"].update(model_config[model_arch])
    model = MaskedAutoencoderViT(**config)

    # Load the data loader
    config = data_config["default"].update(data_config[process_type])
    data_loader = prepare_data(**config)

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define the training loop
    # Define the validation loop






