from os import PathLike
from pathlib import Path
from typing import Union

import torch
from transformers import ViTMAEConfig, ViTMAEForPreTraining, AutoImageProcessor


def rename_key(state_dict):
    """
    Rename keys in state_dict to match the keys in ViTMAEForPreTraining.
    For example, rename "blocks.0.norm1.weight" to "vit.encoder.layer.0.layernorm_before.weight".

    :param state_dict: State dict of our model
    :return: State dict with renamed keys
    """
    # Pair of old name and new name of layers outside of encoder and decoder blocks
    pairs = {
        "cls_token": "vit.embeddings.cls_token",
        "pos_embed": "vit.embeddings.position_embeddings",
        "mask_token": "decoder.mask_token",
        "decoder_pos_embed": "decoder.decoder_pos_embed",
        "patch_embed.proj.weight": "vit.embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.bias": "vit.embeddings.patch_embeddings.projection.bias",

        "norm.weight": "vit.layernorm.weight",
        "norm.bias": "vit.layernorm.bias",
        "decoder_embed.weight": "decoder.decoder_embed.weight",
        "decoder_embed.bias": "decoder.decoder_embed.bias",

        "decoder_norm.weight": "decoder.decoder_norm.weight",
        "decoder_norm.bias": "decoder.decoder_norm.bias",
        "decoder_pred.weight": "decoder.decoder_pred.weight",
        "decoder_pred.bias": "decoder.decoder_pred.bias",
    }

    # Pair of old name and new name of blocks inside of encoder and decoder blocks
    block_type_pairs = {
        "blocks": "vit.encoder.layer",
        "decoder_blocks": "decoder.decoder_layers"
    }

    # Pair of old name and new name of layers inside of encoder and decoder blocks
    layer_type_pairs = {
        "norm1": "layernorm_before",
        "attn.proj": "attention.output.dense",
        "norm2": "layernorm_after",
        "mlp.fc1": "intermediate.dense",
        "mlp.fc2": "output.dense"
    }

    # Pair of old name and new name of layers inside of encoder and decoder blocks.
    # These layers are separated into q, k, v in ViTMAEForPreTraining.
    separare_layer_type_pairs = {
        "attn.qkv": ["attention.attention.query", "attention.attention.key", "attention.attention.value"]
    }

    # Rename keys
    for key in list(state_dict.keys()):
        # Rename layers outside of encoder and decoder blocks
        if key in pairs:
            state_dict[pairs[key]] = state_dict.pop(key)

        # Rename layers inside of encoder and decoder blocks
        else:
            block_type = key.split(".")[0]
            layer_type = ".".join(key.split(".")[2:-1])

            if block_type in block_type_pairs and layer_type in layer_type_pairs:
                new_key = key.replace(block_type, block_type_pairs[block_type])
                new_key = new_key.replace(layer_type, layer_type_pairs[layer_type])

                state_dict[new_key] = state_dict.pop(key)

            elif block_type in block_type_pairs and layer_type in separare_layer_type_pairs:
                # Our model concat q, k, v into one tensor, but ViTMAEForPreTraining has separate tensors for q, k, v
                qkv_tensor = state_dict[key]
                embedding_dim = qkv_tensor.shape[0] // 3
                qkv_list = torch.split(qkv_tensor, embedding_dim)

                state_dict.pop(key)

                for i, new_layer_name in enumerate(separare_layer_type_pairs[layer_type]):
                    new_key = key.replace(block_type, block_type_pairs[block_type])
                    new_key = new_key.replace(layer_type, new_layer_name)

                    state_dict[new_key] = qkv_list[i]

    return state_dict


def get_default_args(default_architecture):
    if default_architecture not in ["mae_vit_base_patch16", "mae_vit_large_patch16", "mae_vit_huge_patch14"]:
        raise ValueError(f"Architecture {default_architecture} not supported")

    if default_architecture == "mae_vit_base_patch16":
        args = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_act": "gelu",  # Unsure
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-6,
            "image_size": 224,
            "patch_size": 16,
            "num_channels": 3,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 512,
            "decoder_num_hidden_layers": 8,
            "norm_pix_loss": False
        }

    elif default_architecture == "mae_vit_large_patch16":
        args = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "hidden_act": "gelu",  # Unsure
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-6,
            "image_size": 224,
            "patch_size": 16,
            "num_channels": 3,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 512,
            "decoder_num_hidden_layers": 8,
            "norm_pix_loss": False
        }

    elif default_architecture == "mae_vit_huge_patch14":
        args = {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "hidden_act": "gelu",  # Unsure
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-6,
            "image_size": 256,  # Unsure
            "patch_size": 14,
            "num_channels": 3,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 512,
            "decoder_num_hidden_layers": 8,
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
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    args = vars(checkpoint['args'])
    weights = checkpoint['model']

    # If default architecture is specified, use default args
    if args.get("architecture", None) is not None:
        args = get_default_args(args['architecture'])

    # Convert config in checkpoint to ViTMAEConfig
    config = ViTMAEConfig(**args)

    # Save config
    config.save_pretrained(model_name, push_to_hub=push_to_hub)

    # Create model ViTMAEForPreTraining with our config
    model = ViTMAEForPreTraining(config)

    # Rename keys to match HuggingFace model
    weights = rename_key(weights)

    # Load weights
    model.load_state_dict(weights)
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
    checkpoint_filename = 'checkpoint(0).pth'
    huggingface_model_name = 'vit-mae-base-1'
    base_processor_name = 'facebook/vit-mae-base'

    current_dir = Path(__file__).parent.resolve()
    ckpt_path = current_dir / 'model' / 'mae' / 'checkpoint' / checkpoint_filename

    convert_to_huggingface_image_processor(base_processor_name, huggingface_model_name, push_to_hub=True)
    convert_to_huggingface_model(ckpt_path, huggingface_model_name, push_to_hub=True)
