import importlib
import os
import sys
import time
from pathlib import Path

import cv2
import imutils
import validators
import wget as wget

from ..models.utils.dataset import get_test_transform
from ..models.utils.output_process import process_output, concat_output
from ..utils.args import show_parameters, remove_parameters, add_parameters
from ..utils.checkpoint import load_checkpoint


def add_infer_arguments(parser):
    """
    Add arguments for inference task

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
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        required=True,
        help="Path to image to infer"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save inference result"
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=("--save" in sys.argv),
        help="Image path for save inference result",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize inference result"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Not print inference result"
    )
    parser.set_defaults(func=infer)


def infer(args):
    ##############################
    # 1. Prepare 'args'
    # Modify to real arguments before showing
    # - Make sure all input and output paths are valid
    # - Make sure checkpoint path is valid (exist path or downloadable url)
    # - Load checkpoint and update model architecture parameters from checkpoint to args
    ##############################

    # Make sure image path is valid
    args.image_path = Path(args.image_path).absolute()
    if not args.image_path.exists():
        raise FileNotFoundError(f'Image file not found: {args.image_path}')

    # Make sure output path is valid
    if args.save:
        args.output_path = Path(args.output_path).absolute()

        # Make sure output directory exists
        os.makedirs(args.output_path.parent, exist_ok=True)

    # Check if checkpoint is a valid path or downloadable url
    if not os.path.exists(args.checkpoint) and not validators.url(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint file not found: {args.checkpoint}')
    elif validators.url(args.checkpoint):
        checkpoint_file = Path(args.checkpoint).name
        checkpoint_path = Path(args.output_path.parent, checkpoint_file).as_posix()

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
        args = add_parameters(args, checkpoint["args"], task="inferring")

    # Finish preparation step by removing unnecessary parameters and show parameters if not no_verbose
    remove_parameters(args)
    if not getattr(args, "no_verbose", False):
        show_parameters(args, "inferring")

    ##############################
    # 2. Prepare model
    # - Load model architecture
    # - Load model weights
    # - Set model to evaluation mode
    ##############################
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

    if not getattr(args, "no_verbose", False):
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    ##############################
    # 3. Inference
    # - Load image
    # - Inference
    # - Save output image if 'save' is True
    # - Visualize output image if 'visualize' is True
    ##############################

    # Load image
    image = cv2.imread(str(args.image_path))

    start_time = time.time()

    # Inference
    input_image = get_test_transform(model=args.model)(image)
    input_image = input_image.unsqueeze(0)

    output = model(input_image, mask_ratio=args.mask_ratio)
    masked, reconstructed, pasted = process_output(image, output, patch_size=model.patch_size)

    if not getattr(args, "no_verbose", False):
        print(f"Inference done in {time.time() - start_time:.2f} seconds")
        print("-" * 60)
        print()

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
