from pathlib import Path

import torch


def save_model(args, epoch, model, optimizer, loss):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f'checkpoint-{epoch}.pth'

    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss.item(),
        'epoch': epoch,
        'args': args,
    }

    torch.save(to_save, checkpoint_path)


def save_best_model(args, model, optimizer, loss):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-best.pth'

    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss.item(),
        'args': args,
    }

    torch.save(to_save, checkpoint_path)


def delete_checkpoint(args, epoch):
    # Delete previous checkpoints, except best and last one
    output_dir = Path(args.output_dir)

    checkpoint_filename = f"checkpoint-{epoch}.pth"
    checkpoint_path = output_dir / checkpoint_filename
    checkpoint_path.unlink(missing_ok=True)
