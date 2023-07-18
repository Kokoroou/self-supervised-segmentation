import pathlib

import torch

if __name__ == "__main__":
    path = "autoencoder/checkpoint/large_epoch75_best.pth"

    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath

        # Load checkpoint
        checkpoint = torch.load(path, map_location="cpu")
    finally:
        pathlib.PosixPath = posix_backup

    print(checkpoint.keys())

    setattr(checkpoint["args"], "wandb_id", "xupn6dzw")

    print(vars(checkpoint["args"]).keys())

    torch.save(checkpoint, path)


