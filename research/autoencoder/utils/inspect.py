from torchinfo import summary


def inspect_model(model, dummy_size):
    """
    Inspect model by summarizing it and printing the total number of parameters.

    :param model: PyTorch model to inspect
    :param dummy_size: Dummy input size.
        E.g: (1, 3, 224, 224) - 1 image with 3 channels, 224x224 pixels
    """
    summary(model, dummy_size)
    print("\nNumber of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

