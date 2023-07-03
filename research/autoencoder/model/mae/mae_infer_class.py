from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from research.autoencoder.model.mae.util.model_process import get_model
from research.autoencoder.model.mae.util.image_process import unpatchify

class AutoencoderClass:
    def __init__(self, model_name: str = "MaskedAutoencoderViT", checkpoint_path: Union[str, Path] = ""):
        """
        Autoencoder class.

        :param model_name: Model name
        :param checkpoint_path: Checkpoint path
        """
        # Check input
        if model_name not in ['MaskedAutoencoderViT']:
            raise ValueError(f'Not supported model: {model_name}')
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')

        # Set attributes
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        # Load model
        if self.model_name == 'MaskedAutoencoderViT':
            self.model = get_model(self.checkpoint_path)

    def __call__(self, image: np.ndarray, mask_ratio: float = 0.75) -> np.ndarray:
        """
        Auto mask and reconstruct image.

        :param image: Image to be processed.
        :param mask_ratio: Masking ratio (percentage of removed patches).
        :return: Reconstructed image.
        """
        return self.inference(image, mask_ratio=mask_ratio)

    def inference(self, image: np.ndarray, mask_ratio: float = 0.75) -> np.ndarray:
        """
        Auto mask and reconstruct image.

        :param image: Image to be processed.
        :param mask_ratio: Masking ratio (percentage of removed patches).
        :return: Reconstructed image.
        """
        image = np.array(image)

        print(image[0][100])

        # Save original shape
        original_shape = image.shape

        # Resize image
        image = cv2.resize(image, (self.model.img_size, self.model.img_size), interpolation=cv2.INTER_AREA)

        # Convert image form shape from (H, W, C) to (B, C, H, W)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)

        # Convert image from numpy to tensor
        image = torch.from_numpy(image).float()

        output = self.model(image, mask_ratio=mask_ratio)

        # Convert image from tensor to numpy
        output = output.detach().numpy()

        # Convert image form shape from (B, C, H, W) to (H, W, C)
        output = output.squeeze()
        output = output.transpose(1, 2, 0)

        # Resize image
        output = cv2.resize(output, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

        return output
