import importlib
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import imutils
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .utils import save_model, save_best_model, delete_checkpoint
from ..models.utils.input_process import process_input
from ..models.utils.output_process import process_output, concat_output


def remove_parameters(args):
    to_delete = []

    for k, v in args.__dict__.items():
        if not v:
            to_delete.append(k)

    for k in to_delete:
        delattr(args, k)


def show_parameters(args, task: str):
    """
    Show the parameters used for training, testing or inferring and remove the ones that are not set
    :param args:
    :param task:
    :return:
    """
    print()
    print("-" * 60)
    print(f"{task.capitalize()} model with:\n")

    for k, v in args.__dict__.items():
        if v and k != "func":
            print(f"{k}={v}")

    print("-" * 60)
    print()


def train(args):
    # Modify to real arguments before showing
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.source_dir = Path(args.source_dir).absolute()
    args.output_dir = Path(args.output_dir).absolute()
    if args.from_pretrained:
        args.checkpoint = Path(args.checkpoint).absolute()

    remove_parameters(args)
    show_parameters(args, "training")

    # Check if paths of source and target directories exist, and checkpoint file exists if from_pretrained is True
    if not args.source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {args.source_dir}')
    if not args.output_dir.exists():
        raise FileNotFoundError(f'Target directory not found: {args.output_dir}')
    if getattr(args, "from_pretrained", False) and not args.checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {args.checkpoint}')

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

    # Move model to device for better performance
    model.to(args.device)

    # Set model to train mode
    model.train()

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

    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = 1.0

    # Create new checkpoint directory for each training
    if hasattr(args, "arch"):
        new_checkpoint_name = f"{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        new_checkpoint_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    new_checkpoint_dir = Path(args.output_dir) / new_checkpoint_name
    args.output_dir = new_checkpoint_dir
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Iterate over the data loader batches
        for inputs, _ in tqdm(list(train_loader)):
            inputs = inputs.to(args.device)

            optimizer.zero_grad()

            # Forward pass
            loss, pred, mask = model(inputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

            save_model(args=args, model=model, optimizer=optimizer, epoch=epoch)

            if loss.item() < best_loss:
                best_loss = loss.item()

                with open(args.output_dir / "info.txt", "w") as f:
                    f.write(f"Best epoch: {epoch}\nLoss: {loss.item()}")

                delete_checkpoint(args=args, epoch="best")

                save_best_model(
                    args=args, model=model, optimizer=optimizer,
                )

            delete_checkpoint(args=args, epoch=epoch - 1)

            inputs.detach()


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

    start_time = time.time()

    # Load model architecture
    module = importlib.import_module("." + args.model, "autoencoder.models")
    if "arch" in args:
        model = getattr(module, args.arch)(**vars(args))
    else:
        model_name = getattr(module, "model_name")
        model = getattr(module, model_name)(**vars(args))

    # Load model weights
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])

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
