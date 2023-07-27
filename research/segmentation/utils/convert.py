from os import PathLike
from pathlib import Path
from typing import Union

from transformers import ViTMAEConfig, AutoImageProcessor

from checkpoint import load_checkpoint
from model import ViTMAESegModel


def get_default_args(default_architecture):
    if default_architecture not in ["vit_mae_seg_base", "vit_mae_seg_large", "vit_mae_seg_huge"]:
        raise ValueError(f"Architecture {default_architecture} not supported")

    if default_architecture == "vit_mae_seg_base":
        args = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,  # Self calculated
            "hidden_act": "gelu",  # Unsure
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-6,
            "image_size": 224,
            "patch_size": 16,
            "num_channels": 3,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 512,
            "decoder_num_hidden_layers": 8,
            "decoder_intermediate_size": 2048,  # Self calculated
            "norm_pix_loss": False
        }

    elif default_architecture == "vit_mae_seg_large":
        args = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,  # Self calculated
            "hidden_act": "gelu",  # Unsure
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-6,
            "image_size": 224,
            "patch_size": 16,
            "num_channels": 3,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 512,
            "decoder_num_hidden_layers": 8,
            "decoder_intermediate_size": 2048,  # Self calculated
            "norm_pix_loss": False
        }

    elif default_architecture == "vit_mae_seg_huge":
        args = {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "intermediate_size": 3136,  # Unsure
            "hidden_act": "gelu",  # Unsure
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-6,
            "image_size": 256,
            "patch_size": 14,
            "num_channels": 3,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 512,
            "decoder_num_hidden_layers": 8,
            "decoder_intermediate_size": 2048,  # Self calculated
            "norm_pix_loss": False
        }
    else:
        raise ValueError(f"Architecture {default_architecture} not supported")

    return args


def convert_to_huggingface_model(checkpoint_path: Union[str, PathLike], model_name: str, push_to_hub: bool = False):
    """
    Convert our model to the HuggingFace model.
    :param checkpoint_path: Path to our model
    :param model_name: Name of the model to save
    :param push_to_hub: Upload model to the HuggingFace model hub
    """
    # Load checkpoint based on platform (Windows or Linux)
    checkpoint = load_checkpoint(checkpoint_path)

    args = vars(checkpoint['args'])
    weights = checkpoint['model']

    # If default architecture is specified, use default args
    if args.get("arch", None) is not None:
        args = get_default_args(args['arch'])

    # Convert config in checkpoint to ViTMAEConfig
    config = ViTMAEConfig(**args)

    # Save config
    config.save_pretrained(model_name, push_to_hub=push_to_hub)

    # Create model ViTMAEForPreTraining with our config
    model = ViTMAESegModel(config)

    # # Rename keys to match HuggingFace model
    # weights = rename_key(weights)

    # Load weights
    model.model.load_state_dict(weights)
    # model.from_pretrained(model_name, state_dict=checkpoint['model'])

    # Save model
    model.save_pretrained(model_name, push_to_hub=push_to_hub)

    return model


def convert_to_huggingface_image_processor(base_name: str, model_name: str, push_to_hub: bool = False):
    """
    Convert our image processor to the HuggingFace image processor.
    :param base_name: Name of the base image processor
    :param model_name: Name of the model that uses this image processor
    :param push_to_hub: Upload image processor to the HuggingFace model hub
    :return: HuggingFace image processor
    """
    image_processor = AutoImageProcessor.from_pretrained(base_name)
    image_processor.save_pretrained(model_name, push_to_hub=push_to_hub)

    return image_processor


if __name__ == '__main__':
    checkpoint_filename = 'base_epoch150_best.pth'
    huggingface_model_name = 'vit-mae-seg-base-polyp'
    base_processor_name = 'facebook/vit-mae-base'
    is_push_to_hub = True

    current_dir = Path(__file__).parent.resolve()
    ckpt_path = current_dir.parent / 'checkpoint' / checkpoint_filename

    convert_to_huggingface_image_processor(base_processor_name, huggingface_model_name, push_to_hub=is_push_to_hub)
    convert_to_huggingface_model(ckpt_path, huggingface_model_name, push_to_hub=is_push_to_hub)
