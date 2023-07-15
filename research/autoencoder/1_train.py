import argparse
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from research.autoencoder.model.mae.mae_architecture import MaskedAutoencoderViT
from research.autoencoder.model.utils.utils import save_model, save_best_model, delete_checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser('MAE training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio (percentage of removed patches)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training / testing")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=int, default=0.001, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Directory for save checkpoint")

    # Model parameters
    parser.add_argument("--img_size", type=int, default=224, help="Image size of input image")
    parser.add_argument("--patch_size", type=int, default=16, help="Size of each patch")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels (e.g. 3 for RGB)")
    parser.add_argument("--encoder_embed_dim", type=int, default=1024, help="Embedding dimension of encoder")
    parser.add_argument("--encoder_depth", type=int, default=24, help="Number of encoder blocks")
    parser.add_argument("--encoder_num_heads", type=int, default=16, help="Number of heads in encoder")
    parser.add_argument("--decoder_embed_dim", type=int, default=512, help="Embedding dimension of decoder")
    parser.add_argument("--decoder_depth", type=int, default=8, help="Number of decoder blocks")
    parser.add_argument("--decoder_num_heads", type=int, default=16, help="Number of heads in decoder")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="Ratio of MLP hidden dim to embedding dim")

    # args = parser.parse_known_args()[0]

    # parser.add_argument('--batch_size', default=32, type=int,
    #                     help='Batch size per GPU')
    # parser.add_argument('--epochs', default=2, type=int)
    # # parser.add_argument('--accum_iter', default=1, type=int,
    # #                     help='Accumulate gradient iterations (for increasing the effective batch size under '
    # #                          'memory constraints)')
    #
    # # Model parameters
    # parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    #
    # parser.add_argument('--input_size', default=224, type=int,
    #                     help='images input size')
    #
    # parser.add_argument('--mask_ratio', default=0.75, type=float,
    #                     help='Masking ratio (percentage of removed patches).')
    #
    # # parser.add_argument('--norm_pix_loss', action='store_true',
    # #                     help='Use (per-patch) normalized pixels as targets for computing loss')
    # # parser.set_defaults(norm_pix_loss=False)
    #
    # # Optimizer parameters
    # # parser.add_argument('--weight_decay', type=float, default=0.05,
    # #                     help='weight decay (default: 0.05)')
    #
    # parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
    #                     help='learning rate (absolute lr)')
    # # parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
    # #                     help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # # parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
    # #                     help='lower lr bound for cyclic schedulers that hit 0')
    #
    # # parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
    # #                     help='epochs to warmup LR')
    #
    # # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    #
    # parser.add_argument('--output_dir', default='./output_dir',
    #                     help='path where to save, empty for no saving')
    # # parser.add_argument('--log_dir', default='./output_dir',
    # #                     help='path where to tensorboard log')
    # # parser.add_argument('--device', default='cuda',
    # #                     help='device to use for training / testing')
    # # parser.add_argument('--seed', default=0, type=int)
    # # parser.add_argument('--resume', default='',
    # #                     help='resume from checkpoint')
    #
    # # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    # #                     help='start epoch')
    # # parser.add_argument('--num_workers', default=10, type=int)
    # # parser.add_argument('--pin_mem', action='store_true',
    # #                     help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # # parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # # parser.set_defaults(pin_mem=True)
    #
    # # distributed training parameters
    # # parser.add_argument('--world_size', default=1, type=int,
    # #                     help='number of distributed processes')
    # # parser.add_argument('--local_rank', default=-1, type=int)
    # # parser.add_argument('--dist_on_itp', action='store_true')
    # # parser.add_argument('--dist_url', default='env://',
    # #                     help='url used to set up distributed training')

    return parser


def main():
    pass


if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()

    parser = get_args_parser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskedAutoencoderViT()
    model.to(device)

    # Define the path to the directory containing the train images
    source_dir = current_dir.parent / "data" / "processed" / "PolypGen2021_MultiCenterData_v3"

    # Define the path to the file containing the train image names
    train_filepath = source_dir / "train_autoencoder.txt"
    test_filepath = source_dir / "test_autoencoder.txt"

    # Open the file with names of training, testing image file, then make DataLoader
    with open(train_filepath, "r") as f:
        train_filenames = f.read().splitlines()
    with open(test_filepath, "r") as f:
        test_filenames = f.read().splitlines()

    # Create a custom dataset class to load the images
    class CustomDataset(ImageFolder):
        def __init__(self, root, names, transform=None):
            super().__init__(root, transform=transform)
            self.samples = [
                (Path(root, name), 0) for name in names
            ]


    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create an instance of the custom dataset
    train_dataset = CustomDataset(str(source_dir), train_filenames, transform=transform)
    test_dataset = CustomDataset(str(source_dir), test_filenames, transform=transform)

    # Create a data loader to load the images in batches
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = 1.0

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Iterate over the data loader batches
        for inputs, _ in tqdm(list(train_loader)):
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

            save_model(args=args, model=model, optimizer=optimizer, epoch=epoch)

            if loss.item() < best_loss:
                best_loss = loss.item()

                with open(Path(current_dir) / "info.txt", "w") as f:
                    f.write(f"Best epoch: {epoch}\nLoss: {loss.item()}")

                delete_checkpoint(args=args, epoch="best")

                save_best_model(
                    args=args, model=model, optimizer=optimizer,
                    criterion=criterion
                )

            delete_checkpoint(args=args, epoch=epoch - 1)
