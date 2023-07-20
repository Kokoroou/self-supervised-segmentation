import cv2
import numpy as np
import torch


def process_input(image, img_size):
    # Imagenet normalization
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    input_image = cv2.resize(image, (img_size, img_size))
    input_image = input_image / 255
    input_image = (input_image - imagenet_mean) / imagenet_std
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
    input_image = torch.from_numpy(input_image)

    return input_image
