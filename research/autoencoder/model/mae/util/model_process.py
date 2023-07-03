import os
import time
from pathlib import Path

import torch

from research.autoencoder.model.mae.mae_architecture import MaskedAutoencoderViT
from research.autoencoder.model.mae import mae_architecture


def get_model(checkpoint_path):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Model structure config get from agrs
    params = ["img_size", "patch_size", "in_chans",
              "encoder_embed_dim", "encoder_depth", "encoder_num_heads",
              "decoder_embed_dim", "decoder_depth", "decoder_num_heads",
              "mlp_ratio"]
    args = vars(checkpoint["args"])

    if args.get("architecture"):
        model = mae_architecture.__dict__[args["architecture"]]()
    else:
        struct = {key: args.get(key) for key in params if args.get(key)}

        # Load model with structure information in args in checkpoint
        model = MaskedAutoencoderViT(**struct)

    model.load_state_dict(checkpoint['model'], strict=False)

    return model


def save_model(args, epoch, model, optimizer, criterion):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f'checkpoint-{epoch}.pth'

    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'epoch': epoch,
        'args': args,
    }

    torch.save(to_save, checkpoint_path)


def save_best_model(args, epoch, model, optimizer, criterion):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f'checkpoint-{epoch}.pth'
    best_checkpoint_path = output_dir / 'checkpoint-best.pth'

    if not checkpoint_path.exists():
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
            'epoch': epoch,
            'args': args,
        }

        torch.save(to_save, checkpoint_path)
    else:
        os.rename(checkpoint_path, best_checkpoint_path)


def delete_checkpoint(args, epoch):
    # Delete previous checkpoints, except best and last one
    output_dir = Path(args.output_dir)

    checkpoint_filename = f"checkpoint-{epoch}.pth"
    checkpoint_path = output_dir / checkpoint_filename
    checkpoint_path.unlink(missing_ok=True)


def save_epoch_result(args, epoch, loss,
                      model, optimizer, criterion):
    global best_loss

    output_dir = Path(args.output_dir)

    start = time.time()
    print(f"Save trained model for epoch {epoch}...")
    save_model(args=args, model=model, optimizer=optimizer,
               criterion=criterion, epoch=epoch)
    print(f"Save trained model for epoch {epoch}. Finish in {time.time() - start}")

    if loss.item() < best_loss:
        best_loss = loss.item()

        with open(Path(output_dir) / "info.txt", "w") as f:
            f.write(f"Best epoch: {epoch}\nLoss: {loss.item()}")

        delete_checkpoint(args=args, epoch="best")

        start = time.time()
        print("Save best trained model...")
        save_best_model(
            args=args, model=model, optimizer=optimizer,
            criterion=criterion, epoch=epoch
        )
        print(f"Save best trained model. Finish in {time.time() - start}")

    delete_checkpoint(args=args, epoch=epoch - 1)




