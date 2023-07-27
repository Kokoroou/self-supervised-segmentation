from transformers import AutoImageProcessor

from ui.models.segmentation.segmentation_result import SegmentorResult
from ui.models.segmentation.vit_seg.vit_seg import ViTMAESegModel


class Segmentor:
    def __init__(self, model):
        """
        Initialize a segmentation model. If 'model' is a checkpoint directory, load the model from the checkpoint.
        :param model: Model name or checkpoint directory
        """
        self.image_processor = AutoImageProcessor.from_pretrained(model)
        self.model = ViTMAESegModel.from_pretrained(model)

    def inference(self, image):
        """
        Perform inference on a single image.
        :param image: Image to perform inference on
        :return: Segmentation mask
        """
        inputs = self.image_processor(images=image, return_tensors="pt")

        outputs = self.model(**inputs)

        return SegmentorResult(image, outputs,
                               image_size=self.model.config.image_size)
