import importlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import validators
import wget
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.utils.dataset import DatasetSegmentation, get_test_transform, get_inverse_test_transform
from ..models.utils.output_process import process_output, concat_output
from ..utils.args import show_parameters, remove_parameters, add_parameters
from ..utils.checkpoint import load_checkpoint
from ..utils.scoring import compute_miou


def add_test_arguments(parser):
    """
    Add arguments for testing task

    :param parser: The parser to add arguments
    """
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Name of the segmentation architecture to use",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to test model. If device is not available, use cpu instead"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save inference result"
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        type=str,
        required=True,
        help="Directory of dataset to test"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=("--save" in sys.argv),
        help="Directory for save test result",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for testing",
    )
    parser.set_defaults(func=test)


def test(args):
    total_time_start = time.time()

    ##############################
    # 1. Prepare 'args'
    # - Make sure all input and output directory are valid
    # - Use the proper device for training ("cuda" or "cpu")
    # - Make sure checkpoint path is valid (exist path or downloadable url),
    # then load checkpoint and update model architecture parameters from checkpoint to args
    ##############################

    # Make sure source directory exists
    args.source_dir = Path(args.source_dir).absolute()
    if not args.source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {args.source_dir}')

    if args.save:
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

    # Load checkpoint based on platform (Windows or Linux)
    checkpoint = load_checkpoint(args.checkpoint)

    # Add parameters from checkpoint to args
    if "args" in checkpoint:
        args = add_parameters(args, checkpoint["args"], task="testing")

    remove_parameters(args)
    show_parameters(args, "testing")

    ##############################
    # 2. Prepare model
    # - Load model architecture
    # - Load model weights
    # - Set model to evaluation mode
    ##############################
    start = time.time()

    # Load model architecture
    module = importlib.import_module("." + args.model, "segmentation.models")
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

    # Set model to evaluation mode
    model.eval()

    print(f"Model loaded in {time.time() - start:.2f} seconds")

    print("-" * 60)
    print()

    ##############################
    # 3. Prepare data
    # - Create custom dataset, apply transformations
    # - Create data loader
    ##############################

    # Create an instance of the custom dataset
    test_dataset = DatasetSegmentation(args.source_dir, transform=get_test_transform(model=args.model))

    # Create a data loader to load the images in batches
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    #############################
    # 4. Test model
    # - Iterate over the data loader batches and test the model
    # - Save test result if 'save' is True
    #############################

    if hasattr(args, "save"):
        for batch_idx, (images, _) in tqdm(enumerate(test_loader), desc=f"Testing {args.model}"):
            # Inferring each image in batch. Need to modify this part to infer all images in batch at once
            for image_idx, image in enumerate(images):
                # Change image from RGB to BGR and add batch dimension
                input_image = torch.flip(image, dims=(0,))
                input_image = input_image.unsqueeze(0)

                # Get original image (resized)
                org_image = get_inverse_test_transform(model=args.model)(image)
                org_image = cv2.cvtColor(np.array(org_image), cv2.COLOR_RGB2BGR)

                output = model(input_image)
                mask = process_output(org_image, output)

                new_image_filename = f"{batch_idx * args.batch_size + image_idx}.jpg"
                new_image_path = args.output_dir / new_image_filename

                cv2.imwrite(str(new_image_path), concat_output(image=org_image, mask=mask))

    else:
        start = time.time()

        # Compute test mIoU
        test_miou = compute_miou(model, test_loader, args.device)

        # Print loss
        print(f"Test mIoU: {test_miou:.4f} - Time: {time.time() - start:.2f} seconds")

    print(f"Testing finished in {time.time() - total_time_start:.2f} seconds")

    print("-" * 60)
    print()
