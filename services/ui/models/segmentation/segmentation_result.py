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

        # Images that can be get from the output object. All images are resized to the original image size.
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
        mask = self.output.detach().cpu()

        mask = torch.round(mask)
        mask = mask[0][0]
        mask = mask.numpy().astype(np.uint8)

        # Resize the mask to the original image size
        mask = cv2.resize(mask, self.input.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # Rescale the segmentation mask to 0-255
        mask_max = mask.max() if mask.max() > 0 else 1
        mask = mask / mask_max * 255

        mask = mask.astype(np.uint8)

        self.segment = mask

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
            if seg_value == 0:
                continue
            seg_color[self.segment == seg_value] = colors[i]

        # Draw the segmentation mask on the input image, with transparency
        seg_color = cv2.addWeighted(self.input, 0.5, seg_color, 0.5, 0)

        self.pasted = seg_color.astype(np.uint8)

        return self.pasted
