import argparse
import importlib
import sys

from ..tasks import add_train_arguments, add_test_arguments, add_infer_arguments


def get_args():
    parser = argparse.ArgumentParser(prog='Segmentation',
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
    # dynamically, because ArgumentParser do not support directly do it. Model's specific arguments cannot show by help
    _args, _ = parser.parse_known_args()

    # Add model's specific arguments
    module = importlib.import_module("." + _args.model, "segmentation.models")
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
