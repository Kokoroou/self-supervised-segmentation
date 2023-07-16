import cv2
import numpy as np
import torch

from ui.models.utils.image_process import unpatchify


class AutoencoderResult:
    def __init__(self, input_img, output, image_size=224, patch_size=16):
        """
        Initialize an output object.
        :param input_img: Input image from the model
        :param output: Output object from the model
        :param image_size: Resized image size
        :param patch_size: Size of the patch
        """
        self.input = np.array(input_img)
        self.output = output
        self.image_size = image_size
        self.patch_size = patch_size

        self.mask = None  # Binary mask. 1 is removing, 0 is keeping
        self.masked = None  # Masked image (input image * mask)
        self.reconstructed = None  # Reconstructed image (output image)
        self.pasted = None  # Pasted image (masked image + reconstructed image)

        self.get_masked_image()
        self.get_reconstructed_image()
        self.get_pasted_image()

    def get_masked_image(self):
        """
        Get the masked image from the output object.
        :return: Masked image
        """
        x = cv2.resize(self.input, (self.image_size, self.image_size))
        x = torch.tensor(x)

        # Visualize the mask
        mask = self.output.mask.detach().cpu()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
        mask = unpatchify(mask, self.patch_size)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask)
        mask = mask[0]

        # Masked image
        masked = x * (1 - mask)

        self.mask = mask.numpy().astype(np.uint8)
        self.masked = masked.numpy().astype(np.uint8)

        return self.masked

    def get_reconstructed_image(self):
        """
        Get the reconstructed image from the output object.
        :return: Reconstructed image
        """
        # Imagenet normalization
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        # Reconstructed image
        reconstructed = self.output.logits.detach().cpu()
        reconstructed = unpatchify(reconstructed, self.patch_size)
        reconstructed = torch.einsum('nchw->nhwc', reconstructed)
        reconstructed = reconstructed[0]
        reconstructed = torch.clip((reconstructed * imagenet_std + imagenet_mean) * 255, 0, 255)

        self.reconstructed = reconstructed.numpy().astype(np.uint8)

        return self.reconstructed

    def get_pasted_image(self):
        """
        Get the pasted image from the output object.
        :return: Pasted image
        """
        self.pasted = self.masked + self.reconstructed * self.mask

        return self.pasted
