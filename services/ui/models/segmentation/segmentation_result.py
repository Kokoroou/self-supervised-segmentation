import cv2
import numpy as np
import torch
from distinctipy import distinctipy


class SegmentorResult:
    def __init__(self, input_img, output, image_size):
        """
        Initialize an output object.
        :param input_img: Input image from the model
        :param output: Output object from the model
        """
        self.input = np.array(input_img)
        self.output = output
        self.image_size = image_size

        self.segment = None  # Segmentation mask
        self.pasted = None  # Pasted image (input image * (1 - segmentation mask) + segmentation mask * 255)

        self.get_segment()
        self.get_pasted_image()

    def get_segment(self):
        """
        Get the segmentation mask from the output object.
        :return: Segmentation mask
        """
        # Get segmentation mask from the output object
        seg = self.output.logits.detach().cpu()

        # Get highest probability for each pixel
        seg = torch.argmax(seg, dim=1)

        # Remove the batch dimension
        seg = torch.einsum('nhw->hw', seg)

        # Resize the segmentation mask to the original image size
        seg = cv2.resize(seg.numpy(), self.input.shape[:2][::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint16)

        # Rescale the segmentation mask to 0-255
        seg = seg / seg.max() * 255

        self.segment = seg.astype(np.uint16)

        return self.segment

    def get_pasted_image(self):
        """
        Get the pasted image from the output object.
        :return: Pasted image
        """
        # Get distinct values in the segmentation mask
        seg_values = np.unique(self.segment)

        # Random color for each distinct value
        colors = np.array(distinctipy.get_colors(len(seg_values))) * 255
        colors = colors.astype(np.uint8)

        # Get the color for each pixel in the segmentation mask
        seg_color = np.zeros((self.segment.shape[0], self.segment.shape[1], 3), dtype=np.uint8)
        for i, seg_value in enumerate(seg_values):
            seg_color[self.segment == seg_value] = colors[i]

        # Draw the segmentation mask on the input image, with transparency
        seg_color = cv2.addWeighted(self.input, 0.5, seg_color, 0.5, 0)

        self.pasted = seg_color.astype(np.uint8)

        return self.pasted
