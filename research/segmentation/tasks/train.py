import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import validators
import wandb
import wget as wget
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.utils.dataset import DatasetSegmentation, get_train_transform, get_test_transform
from ..utils.args import show_parameters, remove_parameters, add_parameters
from ..utils.checkpoint import load_checkpoint, save_model
from ..utils.scoring import compute_miou


def add_train_arguments(parser):
    """
    Add arguments for training task

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
        "--resume",
        action="store_true",
        help="Whether to resume training or not",
    )
    parser.add_argument(
        "--from-pretrained",
        action="store_true",
        help="Whether to load pretrained model or not",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=("--from-pretrained" in sys.argv or "--resume" in sys.argv),
        help="Path to pretrained model",
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
        "--source-dir",
        "-s",
        type=str,
        required=True,
        help="Directory of dataset to train"
    )
    parser.add_argument(
        "--test-dir",
        "-t",
        type=str,
        required=True,
        help="Directory of dataset to test"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./research/segmentation/checkpoint",
        help="Directory for save checkpoint and logging information",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=int,
        default=0.001,
        help="Learning rate"
    )
    parser.set_defaults(func=train)


def train(args):
    total_time_start = time.time()

    ##############################
    # 1. Prepare 'args'
    # - Make sure all input and output directory are valid
    # - Use the proper device for training ("cuda" or "cpu")
    # - If train model from pretrained, make sure checkpoint path is valid (exist path or downloadable url),
    # then load checkpoint and update model architecture parameters from checkpoint to args
    # - Create new output directory based on model architecture name and current datetime
    ##############################

    # Make sure source directory exists
    args.source_dir = Path(args.source_dir).absolute()
    if not args.source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {args.source_dir}')

    # Make sure test directory exists
    args.test_dir = Path(args.test_dir).absolute()
    if not args.test_dir.exists():
        raise FileNotFoundError(f'Test directory not found: {args.test_dir}')

    # Make sure output directory exists
    args.output_dir = Path(args.output_dir).absolute()
    os.makedirs(args.output_dir, exist_ok=True)

    # Make sure device is available
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    checkpoint = None
    if hasattr(args, "checkpoint"):
        # When train from pretrain, check if checkpoint is a valid path or downloadable url
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

        # Load checkpoint based on platform (Windows or Linux)
        checkpoint = load_checkpoint(args.checkpoint)

        # Add parameters from checkpoint to args
        if "args" in checkpoint:
            args = add_parameters(args, checkpoint["args"], task="training" if not args.resume else "resuming")

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
    # - Load model weights if train from pretrained
    # - Set model to train mode
    ##############################
    start = time.time()

    # Load model architecture
    module = importlib.import_module("." + args.model, "segmentation.models")
    if "arch" in args:
        model = getattr(module, args.arch)(**vars(args))
    else:
        model_name = getattr(module, "model_name")
        model = getattr(module, model_name)(**vars(args))

    if hasattr(args, "checkpoint"):
        # Load model weights
        model.load_custom_state_dict(checkpoint["model"])

    # Move model to GPU for better performance
    model.to(args.device)
    # model = nn.DataParallel(model)  # Use all GPUs

    # Set model to train mode
    model.train()

    print(f"Model loaded in {time.time() - start:.2f} seconds")

    print("-" * 60)
    print()

    ##############################
    # 3. Prepare data
    # - Create custom dataset, apply transformations
    # - Create data loader
    ##############################

    # Create an instance of the custom dataset
    train_dataset = DatasetSegmentation(args.source_dir, transform=get_train_transform(model=args.model))
    test_dataset = DatasetSegmentation(args.test_dir, transform=get_test_transform(model=args.model))

    # Create a data loader to load the images in batches
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    #############################
    # 4. Train model
    # - Initialize wandb if enabled
    # - Define optimizer, criterion, best loss
    # - Iterate over the data loader batches and train the model
    # - Save best checkpoint if loss is improved else save last checkpoint
    #############################

    # Initialize wandb
    wandb.init(
        job_type="train",
        dir=args.output_dir,
        config=vars(args),
        project="segmentation",
        id=getattr(args, "wandb_id", None),
        name=args.output_dir.name if getattr(args, "wandb", False) and not hasattr(args, "resume") else None,
        resume="must" if getattr(args, "wandb", False) and hasattr(args, "resume") else None,
        mode="online" if getattr(args, "wandb", False) else "disabled"
    )

    if not hasattr(args, "wandb_id"):
        # Save wandb id to args for later resume training
        setattr(args, "wandb_id", wandb.run.id)

    # Define the optimizer and the loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    # Initialize best loss to a large value and start epoch to 0
    best_loss = 1.0
    start_epoch = 0

    if hasattr(args, "checkpoint") and hasattr(args, "resume"):
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer"])

        best_loss = checkpoint.get("loss", 1.0)

        start_epoch = checkpoint["epoch"] + 1 if "epoch" in checkpoint else 0

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0

        # Iterate over the data loader batches
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            # Move the data to the proper device (GPU or CPU)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # Zero the gradients for every batch
            optimizer.zero_grad()

            # Forward input through the model
            preds = model(inputs)

            # Compute loss
            loss = criterion(preds, labels)

            # Backpropagate loss
            loss.backward()

            # Update weights
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Detach data from GPU
            inputs.detach()
            labels.detach()

        # Compute epoch loss
        current_loss = running_loss / len(train_loader)

        # Compute train mIoU
        train_miou = compute_miou(model, train_loader, args.device)

        # Compute test mIoU
        test_miou = compute_miou(model, test_loader, args.device)

        print(f"Epoch: {epoch}, Loss: {current_loss:.4f}, Train mIoU: {train_miou:.4f}, Test mIoU: {test_miou:.4f}")

        # Log metrics to wandb
        wandb.log({"loss": current_loss, "train_miou": train_miou, "test_miou": test_miou})

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
