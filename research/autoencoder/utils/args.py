from argparse import Namespace


def remove_parameters(args):
    """
    Remove parameters that are not set

    :param args: Arguments
    """
    to_delete = []

    for k, v in vars(args).items():
        if not v:
            to_delete.append(k)

    for k in to_delete:
        delattr(args, k)


def add_parameters(args_1, args_2, task: str = "normal"):
    """
    Add parameters in args_2 that are not set to the ones in args_1

    :param args_1: Arguments
    :param args_2: Arguments
    :param task: Task to perform (normal, training, testing or inferring)
    :return: New arguments with the parameters of args_1 and args_2
    """
    # Parameters that are not used in each task
    filter_out = {
        "normal": ["func"],
        "training": [],
        "testing": [],
        "inferring": ["device", "wandb", "source_dir", "output_dir", "from_pretrained", "batch_size", "epochs",
                      "learning_rate"]
    }

    new_args = vars(args_1).copy()

    for k, v in vars(args_2).items():
        if v and (k not in filter_out[task] and (k not in new_args or not new_args[k])):
            new_args[k] = v

    new_args = Namespace(**new_args)

    return new_args


def show_parameters(args, task: str):
    """
    Show the parameters used for training, testing or inferring and remove the ones that are not set
    :param args: Arguments
    :param task: Task to perform (train, test or infer)
    """
    print()
    print("-" * 60)
    print(f"{task.capitalize()} model with:\n")

    for k, v in vars(args).items():
        if v and k != "func":
            print(f"{k}={v}")

    print("-" * 60)
    print()
