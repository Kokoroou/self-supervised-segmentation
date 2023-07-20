import pathlib
import sys
from pathlib import Path

import torch


def load_checkpoint(checkpoint_path):
    """
    Load checkpoint based on platform

    :param checkpoint_path: Path to checkpoint
    :return: Checkpoint
    """
    # Get platform which is used to load checkpoints
    platform = "Windows" if sys.platform.startswith("win") else "Linux"

    # Load checkpoint on the appropriate platform
    if platform == "Windows":
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        finally:
            pathlib.PosixPath = posix_backup

    elif platform == "Linux":
        windows_backup = pathlib.WindowsPath
        try:
            pathlib.WindowsPath = pathlib.PosixPath

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        finally:
            pathlib.WindowsPath = windows_backup
    else:
        raise Exception(f"Unsupported platform: {platform}")

    return checkpoint


def save_model(args, epoch, model, optimizer, loss, best=False):
    """
    Save model checkpoint

    :param args: Arguments of this run (must include 'output_dir')
    :param epoch: Current epoch number
    :param model: Model to save
    :param optimizer: Optimizer to save
    :param loss: Loss to save
    :param best: Whether this is the best model so far
    """
    # Choose name of checkpoint file
    if best:
        checkpoint_path = Path(args.output_dir, 'best.pth')
    else:
        checkpoint_path = Path(args.output_dir, 'last.pth')

    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'args': args,
    }

    if checkpoint_path.exists():
        delete_checkpoint(args, checkpoint_path.stem)

    torch.save(to_save, checkpoint_path)


def delete_checkpoint(args, name):
    """
    Delete previous checkpoints

    :param args: Arguments (must including 'output_dir')
    :param name: Either 'best' or 'last'
    :return:
    """
    checkpoint_filename = f"{name}.pth"
    checkpoint_path = args.output_dir / checkpoint_filename

    # Delete checkpoint if it exists
    checkpoint_path.unlink(missing_ok=True)
