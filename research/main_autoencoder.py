from autoencoder.utils.dynamic_args_parser import get_args


if __name__ == "__main__":
    args = get_args()
    args.func(args)
