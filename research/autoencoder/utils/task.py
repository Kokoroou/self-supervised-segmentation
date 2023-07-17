import importlib
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import imutils
import torch
import validators
import wandb
import wget as wget
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .args import show_parameters, remove_parameters, add_parameters
from .checkpoint import save_model
from ..models.utils.input_process import process_input
from ..models.utils.output_process import process_output, concat_output


def train(args):
    # Modify to real arguments before showing
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.source_dir = Path(args.source_dir).absolute()
    args.output_dir = Path(args.output_dir).absolute()

    if not args.source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {args.source_dir}')

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.from_pretrained:
        if not os.path.exists(args.checkpoint) and not validators.url(args.checkpoint):
            raise FileNotFoundError(f'Checkpoint file not found: {args.checkpoint}')
        elif validators.url(args.checkpoint):
            checkpoint_file = Path(args.checkpoint).name
            checkpoint_path = Path(args.output_dir, checkpoint_file).as_posix()

            if not Path(checkpoint_path).exists():
                # Download checkpoint file
                print(f"Downloading checkpoint file from {args.checkpoint}")
                wget.download(args.checkpoint, checkpoint_path)

            # Update checkpoint path
            args.checkpoint = checkpoint_path

        args.checkpoint = Path(args.checkpoint).absolute()

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        # Add parameters from checkpoint to args
        if "args" in checkpoint:
            args = add_parameters(args, checkpoint["args"])

    # Create new checkpoint directory for each training
    if hasattr(args, "arch"):
        new_checkpoint_name = f"{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        new_checkpoint_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    new_checkpoint_dir = Path(args.output_dir) / new_checkpoint_name
    args.output_dir = new_checkpoint_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Remove parameters that empty or not used, then show parameters
    remove_parameters(args)
    show_parameters(args, "training")

    #############################
    # Load model
    #############################
    start = time.time()

    # Load model architecture
    module = importlib.import_module("." + args.model, "autoencoder.models")
    if "arch" in args:
        model = getattr(module, args.arch)(**vars(args))
    else:
        model_name = getattr(module, "model_name")
        model = getattr(module, model_name)(**vars(args))

    if getattr(args, "from_pretrained", False):
        # Load model weights
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["model"])

    # Move model to GPU for better performance
    model.to(args.device)
    model = nn.DataParallel(model)  # Use all GPUs

    # Set model to train mode
    model.train()

    print(f"Model loaded in {time.time() - start:.2f} seconds")

    #############################
    # Load data
    #############################
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
    # Train model
    #############################
    wandb.init(
        job_type="train",
        dir=args.output_dir,
        config=vars(args),
        project="autoencoder",
        name=args.output_dir.name,
        mode="online" if getattr(args, "wandb", None) else "disabled"
    )

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.device == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    # Initialize best loss to a large value
    best_loss = 1.0

    for epoch in range(args.epochs):
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
            running_loss += loss.item()

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

    # Finish wandb run
    wandb.finish()


def test(args):
    # Modify to real arguments before showing
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.source_dir = Path(args.source_dir).absolute()
    args.checkpoint = Path(args.checkpoint).absolute()

    remove_parameters(args)
    show_parameters(args, "testing")


def infer(args):
    # Modify to real arguments before showing
    args.checkpoint = Path(args.checkpoint).absolute()
    args.image_path = Path(args.image_path).absolute()
    if args.save:
        args.output_path = Path(args.output_path).absolute()

    remove_parameters(args)

    if not getattr(args, "no-verbose", False):
        show_parameters(args, "inferring")

    # Check if paths of checkpoint and image valid
    if not args.checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {args.checkpoint}')
    if not args.image_path.exists():
        raise FileNotFoundError(f'Image file not found: {args.image_path}')

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Add parameters from checkpoint to args
    if "args" in checkpoint:
        args = add_parameters(args, checkpoint["args"])

    start_time = time.time()

    # Load model architecture
    module = importlib.import_module("." + args.model, "autoencoder.models")
    if "arch" in args:
        model = getattr(module, args.arch)(**vars(args))
    else:
        model_name = getattr(module, "model_name")
        model = getattr(module, model_name)(**vars(args))

    # Load model weights
    model.load_state_dict(checkpoint["model"])

    # Set model to evaluation mode
    model.eval()

    if not getattr(args, "no-verbose", False):
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Load image
    image = cv2.imread(str(args.image_path))

    start_time = time.time()

    # Inference
    input_image = process_input(image, img_size=model.img_size)
    output = model(input_image, mask_ratio=args.mask_ratio)
    masked, reconstructed, pasted = process_output(image, output, patch_size=model.patch_size)

    if not getattr(args, "no-verbose", False):
        print(f"Inference done in {time.time() - start_time:.2f} seconds")

    # Save output image
    if hasattr(args, "save"):
        cv2.imwrite(str(args.output_path), concat_output(image=image, masked=masked,
                                                         reconstructed=reconstructed, pasted=pasted))

    # Visualize output image
    if hasattr(args, "visualize"):
        image = imutils.resize(image, height=700)
        masked = imutils.resize(masked, height=700)
        reconstructed = imutils.resize(reconstructed, height=700)
        pasted = imutils.resize(pasted, height=700)

        # Show output image
        cv2.imshow('image', image)
        cv2.imshow('masked', masked)
        cv2.imshow('reconstructed', reconstructed)
        cv2.imshow('pasted', pasted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
