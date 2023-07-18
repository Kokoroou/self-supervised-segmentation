from pathlib import Path

import torch

from ..utils.args import show_parameters, remove_parameters


def add_test_arguments(parser):
    """
    Add arguments for testing task

    :param parser: The parser to add arguments
    """
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to test model. If device is not available, use cpu instead"
    )
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
        "--source-dir",
        "-s",
        type=str,
        required=True,
        help="Directory of dataset to test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for testing",
    )
    parser.set_defaults(func=test)


def test(args):
    # Modify to real arguments before showing
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.source_dir = Path(args.source_dir).absolute()
    args.checkpoint = Path(args.checkpoint).absolute()

    remove_parameters(args)
    show_parameters(args, "testing")
