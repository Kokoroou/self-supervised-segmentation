from pathlib import Path


def is_checkpoint(path):
    """
    Check if a path is a checkpoint.
    :param path: Path to check
    :return: True if path is a checkpoint, False otherwise
    """
    path = Path(path)

    checkpoint_extensions = [".pt", ".pth", "onnx"]

    if path.is_file() and path.suffix in checkpoint_extensions:
        return True

    return False
