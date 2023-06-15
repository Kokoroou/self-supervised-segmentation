from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms, InterpolationMode


def prepare_data(data_dir: Union[str, Path], transform=None,
                 batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, **kwargs):
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    if transform is None:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    return data_loader


