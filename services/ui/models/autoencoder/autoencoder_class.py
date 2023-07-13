from ui.models.autoencoder.autoencoder_result import AutoencoderResult


class Autoencoder:
    def __init__(self, model):
        """
        Initialize a autoencoder model. If 'model' is a checkpoint path, load the model from the checkpoint.
        :param model: Model name or checkpoint path
        """
        # Public model with checkpoint
        if model in ["facebook/vit-mae-base", "facebook/vit-mae-large", "facebook/vit-mae-huge"]:
            from transformers import AutoImageProcessor, ViTMAEForPreTraining

            self.image_processor = AutoImageProcessor.from_pretrained(model)
            self.model = ViTMAEForPreTraining.from_pretrained(model)

        # # Custom model with checkpoint
        # else:
        #     # Simple converter to convert image to tensor
        #     self.image_processor = None
        #     if is_checkpoint(model):
        #         args = torch.load(model, map_location=torch.device('cpu'))['args']

    def inference(self, image, mask_ratio: float = 0.75):
        """
        Perform inference on a single image.

        :param image: Image to perform inference on
        :param mask_ratio: Ratio of the image to be masked
        :return: Output object that contains the reconstructed image, mask, and loss
        """
        inputs = self.image_processor(images=image, return_tensors="pt")

        self.model.config.mask_ratio = mask_ratio

        outputs = self.model(**inputs)

        return AutoencoderResult(image, outputs,
                                 image_size=self.model.config.image_size,
                                 patch_size=self.model.config.patch_size)
