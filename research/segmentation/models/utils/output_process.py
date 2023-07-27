import cv2
import numpy as np
import torch


def process_output(image, output):
    # Visualize the mask
    mask = output.detach().cpu()
    mask = torch.round(mask)
    mask = mask[0][0]
    mask = mask.numpy().astype(np.uint8)

    # Resize the mask to the original image size
    mask = cv2.resize(mask, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    # Rescale the segmentation mask to 0-255
    mask_max = mask.max() if mask.max() > 0 else 1
    mask = mask / mask_max * 255

    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return mask


def concat_output(**kwargs):
    """
    Concatenate all images into one by horizontally stacking them.

    :param kwargs: Dictionary of images to concatenate. Keys are the names of the images,
        and values are the images themselves. Images must be numpy arrays and have the same height, channel.
    :return: Concatenated image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    padding = 10  # Padding between image and text

    images = []
    for name, image in kwargs.items():
        if image is not None:
            # Get the width and height of the image
            image_height, image_width, image_channel = image.shape

            # Get the relative font scale and thickness based on the image size
            font_scale = image_height / 256
            thickness = round(font_scale)

            # Get the width and height of the text box
            text_width, text_height = cv2.getTextSize(name, font, font_scale, thickness)[0]

            # Get coordinates based on the text size
            text_x = (image_width - text_width) // 2
            text_y = image_height + text_height + padding

            # Create new image to put the text out of the image.
            # Add 0.5 text height for the tail of some characters (p, g, j, ...)
            new_image = np.zeros((image_height + int(1.5 * text_height) + padding, image_width, image_channel),
                                 dtype=np.uint8)

            # Put the image and text into the new image
            new_image[:image.shape[0], :, :] = image
            new_image = cv2.putText(new_image, name, (text_x, text_y), font, font_scale, color, thickness)

            # Add the new image to the list
            images.append(new_image)

            # Get the size of padding between this image and the next image
            padding_channel = new_image.shape[2]
            padding_height = new_image.shape[0]
            padding_width = padding_height // 10

            # Add the padding
            images.append(np.zeros((padding_height, padding_width, padding_channel), dtype=np.uint8))

    # Remove the last padding
    images.pop()

    return np.hstack(images)

