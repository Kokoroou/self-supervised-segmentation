class Detector:
    def __init__(self, model):
        """
        Initialize a detection model. If 'model' is a checkpoint path, load the model from the checkpoint.
        :param model: Model name or checkpoint path
        """
        pass

    def load(self, checkpoint_path):
        """
        Load a model from a checkpoint path.
        :param checkpoint_path: Path to the checkpoint
        """
        pass

    def inference(self, image):
        """
        Perform inference on a single image.
        :param image: Image to perform inference on
        :return: List of bounding boxes
        """
        pass
