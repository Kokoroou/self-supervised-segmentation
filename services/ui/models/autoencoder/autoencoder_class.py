import numpy as np
import torch

from ui.models.utils.checker import is_checkpoint
from ui.models.utils.image_process import unpatchify


class Autoencoder:
    def __init__(self, model):
        """
        Initialize a autoencoder model. If 'model' is a checkpoint path, load the model from the checkpoint.
        :param model: Model name or checkpoint path
        """
        if model == "facebook/vit-mae-base":
            from transformers import AutoImageProcessor, ViTMAEModel, ViTMAEForPreTraining

            self.image_processor = AutoImageProcessor.from_pretrained(model)
            self.model = ViTMAEForPreTraining.from_pretrained(model)

        elif is_checkpoint(model):
            self.model = None

            # Load model with checkpoint
            self.load(model)

    def load(self, checkpoint_path):
        """
        Load a custom model from a checkpoint path.
        :param checkpoint_path: Path to the checkpoint
        """
        pass

    def inference(self, image):
        """
        Perform inference on a single image.
        :param image: Image to perform inference on
        :return: Output object that contains the reconstructed image, mask, and loss
        """
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        logits = outputs.logits

        # print(logits.shape)
        # print(type(logits))
        # print(logits)

        logits = unpatchify(logits, 16)
        # print(logits.shape)

        logits = torch.einsum('nchw->nhwc', logits).detach().cpu()[0]
        # print(logits.shape)

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        logits = torch.clip((logits * imagenet_std + imagenet_mean) * 255, 0, 255)

        logits = logits.numpy().astype(np.uint8)

        return logits
