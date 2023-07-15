from pathlib import Path

import torch

if __name__ == "__main__":
    checkpoint_dir = Path(__file__).parent.resolve()
    checkpoint_path = checkpoint_dir / 'checkpoint(0).pth'

    # Open an pytorch checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # # Print the keys of the checkpoint
    # print(checkpoint['args'].asfj)

    args = vars(checkpoint["args"])

    if args.get("asfj"):
        print("True")
    else:
        print("False")

    # # Update key args of the checkpoint (args is a Namespace object)
    # checkpoint['args'].architecture = "mae_vit_base_patch16"
    #
    # # Save the updated checkpoint
    # torch.save(checkpoint, checkpoint_path)
    