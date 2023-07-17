from argparse import Namespace
from pathlib import Path

import torch

if __name__ == "__main__":
    checkpoint_dir = Path(__file__).parent.resolve()
    checkpoint_path = checkpoint_dir / 'mae_visualize_vit_large_ganloss.pth'
    checkpoint_architecture = "mae_vit_large_patch16"

    # Open an pytorch checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if "args" not in checkpoint:
        # Add a Namespace object to the checkpoint
        checkpoint["args"] = Namespace()

    args = vars(checkpoint["args"])

    # Update key args of the checkpoint (args is a Namespace object)
    checkpoint['args'].architecture = checkpoint_architecture

    # Save the updated checkpoint
    torch.save(checkpoint, checkpoint_path)
    