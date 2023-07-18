import importlib
import os
import pathlib
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import validators
import wandb
import wget as wget
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from ..utils.args import show_parameters, remove_parameters, add_parameters
from ..utils.checkpoint import save_model


platform = "Windows" if sys.platform.startswith("win") else "Linux"


def add_resume_arguments(parser):
    """
    Add arguments for resume training task

    :param parser: The parser to add arguments
    """
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Name of the autoencoder architecture to use",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to pretrained model for resume training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train model. If device is not available, use cpu instead"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb to log training information or not"
    )
    parser.add_argument(
        "--wandb-id",
        type=str,
        required=("--wandb" in sys.argv),
        help="ID of wandb run to resume"
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        type=str,
        help="Directory of dataset to train"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Directory for save checkpoint and logging information",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs. Training will resume from last epoch in checkpoint"
    )
    parser.add_argument(
        "--learning-rate",
        type=int,
        help="Learning rate"
    )
    parser.set_defaults(func=resume)


def resume(args):
    total_time_start = time.time()

    ##############################
    # 1. Prepare 'args'
    # - Make sure all input and output directory are valid
    # - Use the proper device for training ("cuda" or "cpu")
    # - Make sure checkpoint path is valid (exist path or downloadable url),
    # then load checkpoint and update model architecture parameters from checkpoint to args
    # - Create new output directory based on model architecture name and current datetime
    ##############################

    # Make sure source directory exists
    args.source_dir = Path(args.source_dir).absolute()
    if not args.source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {args.source_dir}')

    # Make sure output directory exists
    args.output_dir = Path(args.output_dir).absolute()
    os.makedirs(args.output_dir, exist_ok=True)

    # Make sure device is available
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Check if checkpoint is a valid path or downloadable url
    if not os.path.exists(args.checkpoint) and not validators.url(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint file not found: {args.checkpoint}')
    elif validators.url(args.checkpoint):
        checkpoint_file = Path(args.checkpoint).name
        checkpoint_path = Path(args.output_dir, checkpoint_file).as_posix()

        # Check if checkpoint file downloaded
        if not Path(checkpoint_path).exists():
            # Download checkpoint file
            print(f"Downloading checkpoint file from {args.checkpoint}")
            wget.download(args.checkpoint, checkpoint_path)

        # Update checkpoint path
        args.checkpoint = checkpoint_path
    args.checkpoint = Path(args.checkpoint).absolute()

    if platform == "Windows":
        # Resume on Windows
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath

            # Load checkpoint
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
        finally:
            pathlib.PosixPath = posix_backup

    elif platform == "Linux":
        # Resume on Linux
        windows_backup = pathlib.WindowsPath
        try:
            pathlib.WindowsPath = pathlib.PosixPath

            # Load checkpoint
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
        finally:
            pathlib.WindowsPath = windows_backup
    else:
        raise Exception(f"Unsupported platform: {platform}")

    # Add parameters from checkpoint to args
    if "args" in checkpoint:
        args = add_parameters(args, checkpoint["args"], task="resuming")

    # Create new checkpoint directory for each training
    if hasattr(args, "arch"):
        new_checkpoint_name = f"{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        new_checkpoint_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Update output directory in 'args'
    new_checkpoint_dir = Path(args.output_dir) / new_checkpoint_name
    os.makedirs(new_checkpoint_dir, exist_ok=True)
    args.output_dir = new_checkpoint_dir

    # Finish preparation step by remove parameters that empty or not used, then show parameters
    remove_parameters(args)
    show_parameters(args, "training")

    ##############################
    # 2. Prepare model
    # - Load model architecture
    # - Load model weights
    # - Set model to train mode
    ##############################
    start = time.time()

    # Load model architecture
    module = importlib.import_module("." + args.model, "autoencoder.models")
    if "arch" in args:
        model = getattr(module, args.arch)(**vars(args))
    else:
        model_name = getattr(module, "model_name")
        model = getattr(module, model_name)(**vars(args))

    # Load model weights
    model.load_state_dict(checkpoint["model"])

    # Move model to GPU for better performance
    model.to(args.device)
    # model = nn.DataParallel(model)  # Use all GPUs

    # Set model to train mode
    model.train()

    print(f"Model loaded in {time.time() - start:.2f} seconds\n")

    ##############################
    # 3. Prepare data
    # - Create custom dataset, apply transformations
    # - Create data loader
    ##############################

    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create an instance of the custom dataset
    train_dataset = ImageFolder(str(args.source_dir), transform=transform)

    # Create a data loader to load the images in batches
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    #############################
    # 4. Train model
    # - Initialize wandb if enabled
    # - Define optimizer, scaler, best loss
    # - Iterate over the data loader batches and train the model
    # - Save best checkpoint if loss is improved else save last checkpoint
    #############################

    wandb.init(
        job_type="train",
        dir=args.output_dir,
        config=vars(args),
        project="autoencoder",
        id=getattr(args, "wandb_id", None),
        resume="must",
        mode="online" if getattr(args, "wandb", None) else "disabled"
    )

    print()
    print("-" * 60)
    print()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer"])

    scaler = torch.cuda.amp.GradScaler(enabled=True if args.device == "cuda" else False)

    # Initialize best loss to a large value
    best_loss = checkpoint.get("loss", 1.0)

    start_epoch = checkpoint["epoch"] + 1 if "epoch" in checkpoint else 0

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0

        # Iterate over the data loader batches
        for inputs, _ in tqdm(train_loader):
            # Move the data to the proper device (GPU or CPU)
            inputs = inputs.to(args.device)

            # Zero the gradients for every batch
            optimizer.zero_grad()

            # Forward input through the model
            loss, pred, mask = model(inputs)

            # Scale Gradients: Compute the loss's gradients
            scaler.scale(loss).backward()

            # Update Optimizer: Adjust learning weights
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss
            running_loss += loss.mean().item()

            # Detach inputs from GPU
            inputs.detach()

        # Compute epoch loss
        current_loss = running_loss / len(train_loader)
        print(f"Epoch: {epoch}, Loss: {current_loss:.4f}")

        # Log metrics to wandb
        wandb.log({"loss": current_loss})

        # Check if current loss is better than best loss
        if current_loss < best_loss:
            best = True
            best_loss = current_loss

            with open(args.output_dir / "info.txt", "w") as f:
                f.write(f"Best epoch: {epoch}\nLoss: {current_loss}")
        else:
            best = False

        # Save checkpoint
        save_model(args=args, epoch=epoch, model=model, optimizer=optimizer, loss=current_loss, best=best)

    print("-" * 60)
    print()

    wandb.alert(
        title="Training finished",
        text=f"Training finished in {time.time() - total_time_start:.2f} seconds",
        level=wandb.AlertLevel.INFO
    )

    # Finish wandb run
    wandb.finish()
