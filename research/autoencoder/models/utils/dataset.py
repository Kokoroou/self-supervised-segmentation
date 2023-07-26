from torchvision import transforms


def get_train_transform(model: str):
    """
    Get transform for training

    :param model: Name of the model
    :return: Transform for training
    """

    if model == "mae":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = None

    return transform


def get_test_transform(model: str):
    """
    Get transform for testing

    :param model: Name of the model
    :return: Transform for testing
    """

    if model == "mae":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = None

    return transform


def get_inverse_test_transform(model: str):
    """
    Get inverse transform for testing

    :param model: Name of the model
    :return: Inverse transform for testing
    """

    if model == "mae":
        transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.ToPILImage()
        ])
    else:
        transform = None

    return transform
