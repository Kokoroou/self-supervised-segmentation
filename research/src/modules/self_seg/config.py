from functools import partial

import torch.nn as nn
from torchvision.transforms import transforms, InterpolationMode

model_config = {
    "default": {
        "img_size": 224,  # Input image size (both width and height)
        "patch_size": 16,  # Patch size (both width and height)
        "in_chans": 3,  # Number of input channels (RGB has 3)
        "embed_dim": 1024,  # Embedding dimension (number of features) of the encoder
        "depth": 24,  # Depth of the network (number of blocks) of the encoder
        "num_heads": 16,  # Number of head in multi-head attention of the encoder

        "decoder_embed_dim": 512,  # Embedding dimension (number of features) of the decoder
        "decoder_depth": 8,  # Depth of the decoder
        "decoder_num_heads": 16,  # Number of head in multi-head attention of the decoder

        "mlp_ratio": 4.,  # Ratio of Multi-Layer Perceptron expansion in each block
        "norm_layer": nn.LayerNorm,  # Normalization layer
        "norm_pix_loss": False  # Normalize pixel loss with number of pixels in image (True) or not (False)
    },
    "mae_vit_base_patch16": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    },
    "mae_vit_large_patch16": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    },
    "mae_vit_huge_patch14": {
        "patch_size": 14,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    },
}

data_config = {
    "default": {
        "dataset": "imagenet",
        "data_dir": "data/imagenet",
        "transform": transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 4,

        "pin_memory": True,
        "drop_last": True,

        "distributed": False,
        "mixed_precision": False,
        "amp_level": "O0",
        "num_samples": None,
        "num_classes": 1000,
        "crop_pct": 1.0,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_patches": None,
        "mask_ratio": 0.75,
        "mask_size": 16,
        "mask_prob": 0.15,
        "mask_overlap": 0.5,
        "mask_noise": 0.03,
        "mask_use_mean": False,
        "mask_start_idx": 0,
        "mask_init": "zeros",
        "mask_init_prob": 0.5,
        "mask_init_range": (0., 0.),
        "mask_train": True
    },
}

logging_config = {
    "default": {
        "log_dir": "logs",
        "log_freq": 10,
        "save_freq": 10,
        "save_dir": "checkpoints",
        "save_name": "mae_vit_base_patch16.pth",
        "save_last": True,
        "save_best": True,
    },
}

training_config = {
    "default": {
        "epochs": 100,
        "lr": 0.001,
        "lr_decay": 0.1,
        "lr_decay_epochs": [30, 60, 90],
        "weight_decay": 0.05,
        "momentum": 0.9,
        "nesterov": False,
        "optimizer": "adam",
        "scheduler": "cosine",
        "warmup_epochs": 10,
        "warmup_lr": 0.0001,
        "warmup_method": "linear",
        "clip_grad": 0.0,
        "grad_norm": None,
        "grad_norm_type": 2,
        "seed": 42,
        "log_level": "info",
        "log_interval": 10,
        "save_interval": 10,
        "save_dir": "checkpoints",
        "save_name": "mae_vit_base_patch16.pth",
        "save_last": True,
        "save_best": True,
        "resume": None,
        "amp_level": "O0",
        "num_workers": 4,
        "pin_memory": True,
        "distributed": False,
        "mixed_precision": False,
        "device": "cuda",
        "use_ddp": False,
        "use_dali": False,
        "use_apex": False,
        "use_amp": False,
        "use_tfboard": False,
        "use_tensorboard": False,
        "use_wandb": False,
        "use_swa": False,
        "use_ema": False,
        "ema_decay": 0.9999,
        "ema_start": 0,
        "ema_warmup": False,
        "ema_warmup_epochs": 10,
        "ema_warmup_decay": 0.9,
        "ema_eval_freq": 1,
        "ema_eval_start": 0,
        "ema_eval_best": False,
        "ema_eval_metric": "loss",
        "ema_eval_metric_mode": "min",
    },
}

test_config = {
    "default": {
        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True,
        "shuffle": False,
        "distributed": False,
        "mixed_precision": False,
        "amp_level": "O0",
        "num_samples": None,
        "num_classes": 1000,
        "crop_pct": 1.0,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_patches": None,
        "mask_ratio": 0.75,
        "mask_size": 16,
        "mask_prob": 0.15,
        "mask_overlap": 0.5,
        "mask_noise": 0.03,
        "mask_use_mean": False,
        "mask_start_idx": 0,
        "mask_init": "zeros",
        "mask_init_prob": 0.5,
        "mask_init_range": (0., 0.),
        "mask_train": False
    },
}
