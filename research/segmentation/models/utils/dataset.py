import random
from pathlib import Path

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, transform=None):
        super(DatasetSegmentation, self).__init__()

        self.transform = transform

        folder_path = Path(folder_path)

        self.img_files = list((folder_path / 'images').glob('*.*'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(folder_path / 'masks' / img_path.name)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = Image.open(str(img_path))
        label = Image.open(str(mask_path))

        if self.transform:
            data = self.transform(data)

            label_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(data.shape[1:], interpolation=Image.NEAREST),
            ])

            label = label_transform(label)

            random_transform = random.choice([
                transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(0, 0)),  # No change
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation([90, 90]),
                transforms.RandomRotation([180, 180]),
                transforms.RandomRotation([270, 270]),
            ])

            data = random_transform(data)
            label = random_transform(label)

            cv2.imshow("data", data.permute(1, 2, 0).numpy())
            cv2.imshow("label", label.permute(1, 2, 0).numpy())
            cv2.waitKey(0)

        return data, label

    def __len__(self):
        return len(self.img_files)


def get_train_transform(model: str):
    """
    Get transform for training

    :param model: Name of the model
    :return: Transform for training
    """

    if model == "vit_unetr":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=(0, 0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = None

    return transform


def get_test_transform(model: str):
    """
    Get transform for testing

    :param model: Name of the model
    :return: Transform for testing
    """

    if model == "vit_unetr":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = None

    return transform


def get_inverse_test_transform(model: str):
    """
    Get inverse transform for testing

    :param model: Name of the model
    :return: Inverse transform for testing
    """

    if model == "vit_unetr":
        transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.ToPILImage()
        ])
    else:
        transform = None

    return transform
