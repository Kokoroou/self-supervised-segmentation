from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
import cv2

import models_mae


# define the utils

# def show_image(image, title=''):
#     # image is [H, W, 3]
#     assert image.shape[2] == 3
#     plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title(title, fontsize=16)
#     plt.axis('off')
#     return


# def run_one_image(img, model):
#     x = torch.tensor(img)
#
#     # make it a batch-like
#     x = x.unsqueeze(dim=0)
#     x = torch.einsum('nhwc->nchw', x)
#
#     # run MAE
#     loss, y, mask = model(x.float(), mask_ratio=0.75)
#     y = model.unpatchify(y)
#     y = torch.einsum('nchw->nhwc', y).detach().cpu()
#
#     # visualize the mask
#     mask = mask.detach()
#     mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
#     mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
#     mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
#
#     x = torch.einsum('nchw->nhwc', x)
#
#     # masked image
#     im_masked = x * (1 - mask)
#
#     # MAE reconstruction pasted with visible patches
#     im_paste = x * (1 - mask) + y * mask
#
#     # make the plt figure larger
#     plt.rcParams['figure.figsize'] = [24, 24]
#
#     plt.subplot(1, 4, 1)
#     show_image(x[0], "original")
#
#     plt.subplot(1, 4, 2)
#     show_image(im_masked[0], "masked")
#
#     plt.subplot(1, 4, 3)
#     show_image(y[0], "reconstruction")
#
#     plt.subplot(1, 4, 4)
#     show_image(im_paste[0], "reconstruction + visible")
#
#     plt.show()


class MAE:
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    def __init__(self, architecture: str, checkpoint_dir: str):
        if architecture != 'mae_vit_base_patch16':
            raise NotImplementedError(f'Only mae_vit_base_patch16 is supported, not {architecture}')

        self.architecture = architecture  # Name of architecture or path to model
        self.checkpoint_dir = checkpoint_dir  # Path to checkpoint

        self.model = models_mae.__dict__[self.architecture]()
        self.model.load_state_dict(torch.load(self.checkpoint_dir, map_location='cpu')['model'])

    def __call__(self, image, mask_ratio: float = 0.75, *args, **kwargs):
        original_shape = image.size

        image = image.resize((224, 224))
        image = np.array(image) / 255.

        x = torch.tensor(image)

        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        # run MAE
        loss, y, mask = self.model(x.float(), mask_ratio=mask_ratio)
        y = self.model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask = self.model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', x)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        im_masked = im_masked[0].numpy()
        y = y[0].numpy()
        im_paste = im_paste[0].numpy()

        im_masked = im_masked * 255
        y = y * 255
        im_paste = im_paste * 255

        im_masked = np.clip(im_masked, 0, 255).astype(np.uint8)
        y = np.clip(y, 0, 255).astype(np.uint8)
        im_paste = np.clip(im_paste, 0, 255).astype(np.uint8)

        im_masked = cv2.resize(im_masked, original_shape)
        im_reconstructed = cv2.resize(y, original_shape)
        im_paste = cv2.resize(im_paste, original_shape)

        return im_masked, im_reconstructed, im_paste
        # return self.masking(image, mask_ratio), self.reconstruct(image), self.show_final(image)

    def prepare_model(self):
        # build model
        model = getattr(models_mae, self.architecture)()

        # load model
        checkpoint = torch.load(self.checkpoint_dir, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)

        return model

    def masking(self, image, mask_ratio: float = 0.75) -> np.ndarray:
        pass

    def reconstruct(self, image) -> np.ndarray:
        pass

    def show_final(self, image) -> np.ndarray:
        pass


if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    main_dir = current_dir.parent.parent.parent
    image_path = main_dir / 'research' / 'mae' / 'dataset' / 'train' / 'images_C1' / '100S0001.jpg'
    checkpoint_path = main_dir / 'research' / 'mae' / 'output_dir' / 'checkpoint-best.pth'

    image = Image.open(image_path)

    mae = MAE('mae_vit_base_patch16', str(checkpoint_path))

    im_masked, im_reconstructed, im_paste = mae(image)

    im_masked = cv2.cvtColor(im_masked, cv2.COLOR_RGB2BGR)
    im_reconstructed = cv2.cvtColor(im_reconstructed, cv2.COLOR_RGB2BGR)
    im_paste = cv2.cvtColor(im_paste, cv2.COLOR_RGB2BGR)

    cv2.imshow('im_masked', im_masked)
    cv2.imshow('im_reconstructed', im_reconstructed)
    cv2.imshow('im_paste', im_paste)
    cv2.waitKey(0)
