from statistics import mean
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm


def compute_miou(model, data_loader, device):
    """
    Compute mean intersection over union (mIoU) for a model

    Args:
        model: model to compute mIoU
        data_loader: DataLoader for evaluate model
        device: device to use for evaluate model
    Returns:
        miou: mean intersection over union score
    """
    # Set model to evaluate mode
    model.eval()

    # Initialize list of intersection over union
    ious = []

    # Iterate over batches
    for images, masks in tqdm(data_loader, desc="Computing mIoU"):
        # Move images and masks to device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        masks = torch.round(masks).to(torch.int64)
        outputs = torch.round(outputs).to(torch.int64)

        batch_size = images.size()[0]

        # Calculate intersection over union
        iou = mean(calculate_intersection_over_union(masks[i][0], outputs[i][0]) for i in range(batch_size))

        # Append to list of intersection over union
        ious.append(iou)

    # Calculate mean intersection over union
    miou = mean(ious)

    return miou


def calculate_intersection_over_union(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Calculate intersection over union

    Args:
        y_true: 2D array of ground truth labels
        y_pred: 2D array of predicted labels
    Returns:
        iou: intersection over union score
    """
    if y_true.size() != y_pred.size():
        raise ValueError(f"y_true and y_pred must have the same shape: {y_true.size()} != {y_pred.size()}")
    if len(y_true.size()) != 2:
        raise ValueError(f"y_true and y_pred must be 2D: {len(y_true.size())} != 2")

    # Calculate intersection and union for each class
    intersection = torch.logical_and(y_true, y_pred)
    union = torch.logical_or(y_true, y_pred)

    # Sum the number of True values in each intersection and union
    intersection = torch.sum(intersection)
    union = torch.sum(union)

    # Calculate intersection over union for each class
    iou = intersection / union

    return iou.item()


def calculate_pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Calculate pixel accuracy

    Args:
        y_true: 2D array of ground truth labels
        y_pred: 2D array of predicted labels
    Returns:
        pixel_accuracy: pixel accuracy score
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape: {y_true.shape} != {y_pred.shape}")
    if len(y_true.shape) != 2:
        raise ValueError(f"y_true and y_pred must be 2D: {len(y_true.shape)} != 2")

    # Calculate pixel accuracy
    pixel_accuracy = np.sum(y_true == y_pred) / y_true.size

    return pixel_accuracy


def calculate_dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Calculate dice coefficient

    Args:
        y_true: 2D array of ground truth labels
        y_pred: 2D array of predicted labels
    Returns:
        dice_coefficient: dice coefficient score
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape: {y_true.shape} != {y_pred.shape}")
    if len(y_true.shape) != 2:
        raise ValueError(f"y_true and y_pred must be 2D: {len(y_true.shape)} != 2")

    # Calculate intersection and union for each class
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)

    # Sum the number of True values in each intersection and union
    intersection = np.sum(intersection)
    union = np.sum(union)

    # Calculate dice coefficient for each class
    dice_coefficient = 2 * intersection / (intersection + union)

    return dice_coefficient


def calculate_score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.float64, np.float64, np.float64]:
    """
    Calculate IoU, pixel accuracy, and dice coefficient

    Args:
        y_true: 2D array of ground truth labels
        y_pred: 2D array of predicted labels
    Returns:
        iou: intersection over union score
        pixel_accuracy: pixel accuracy score
        dice_coefficient: dice coefficient score
    """
    return calculate_intersection_over_union(y_true, y_pred), \
           calculate_pixel_accuracy(y_true, y_pred), \
           calculate_dice_coefficient(y_true, y_pred)


if __name__ == "__main__":
    y_true_demo = np.array([[0, 1, 1, 0],
                            [1, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 0, 1]])
    y_pred_demo = np.array([[1, 1, 1, 0],
                            [1, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 0, 1]])
    # y_pred_demo = np.ones(shape=y_pred_demo.shape, dtype=np.int32) - y_pred_demo

    iou_demo = calculate_score(y_true_demo, y_pred_demo)
    print(iou_demo)
    print(type(iou_demo))

