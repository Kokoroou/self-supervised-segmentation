import argparse
import importlib
import sys

from .task import train, test, infer


def add_train_arguments(parser):
    """
    Add arguments for training task

    :param parser: The parser to add arguments
    """
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train model. If device is not available, use cpu instead"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Name of the autoencoder architecture to use",
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        type=str,
        required=True,
        help="Directory of dataset to train"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./autoencoder/checkpoint",
        help="Directory for save checkpoint and logging information",
    )
    parser.add_argument(
        "--from-pretrained",
        action="store_true",
        help="Whether to load pretrained model or not",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=("--from-pretrained" in sys.argv),
        help="Path to pretrained model",
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
        type=str,
        required=("--save" in sys.argv),
        help="Directory for save inference result",
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


def get_args():
    parser = argparse.ArgumentParser(prog='Autoencoder',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="Sub-commands help")

    # Create subparsers for each task
    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_test = subparsers.add_parser("test", help="Test a model")
    parser_infer = subparsers.add_parser("infer", help="Infer a model on one image")

    # Add arguments for each task
    add_train_arguments(parser_train)
    add_test_arguments(parser_test)
    add_infer_arguments(parser_infer)

    # Parse arguments first time to get model name. This is a tricky method to add model's specific arguments
    # dynamically, because ArgumentParser do not support directly do it.
    _args, _ = parser.parse_known_args()

    # Add model's specific arguments
    module = importlib.import_module("." + _args.model, "autoencoder.models")
    add_model_arguments = getattr(module, "add_model_arguments")
    if "train" in sys.argv:
        add_model_arguments(parser_train)
    elif "test" in sys.argv:
        add_model_arguments(parser_test)
    elif "infer" in sys.argv:
        add_model_arguments(parser_infer)

    # Parse arguments second time to get all arguments and return
    args = parser.parse_args()

    return args
