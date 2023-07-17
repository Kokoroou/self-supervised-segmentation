import random
import shutil
from pathlib import Path, PurePosixPath

from tqdm import tqdm


def split_data(data_dir, train_ratio: float = 0.8, seed: int = 42):
    """
    Load list of image file names, then split it into train and test sets and write them to a text file.

    :param data_dir: Path to directory containing images
    :param train_ratio: Ratio of train set to total set
    :param seed: Random seed
    :return:
    """
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError("train_ratio must be between 0 and 1")

    source_dir = data_dir / "processed" / "PolypGen2021_MultiCenterData_v3"

    train_autoencoder_file = source_dir / "train_autoencoder.txt"
    test_autoencoder_file = source_dir / "test_autoencoder.txt"
    train_segmentation_file = source_dir / "train_segmentation.txt"
    test_segmentation_file = source_dir / "test_segmentation.txt"

    autoencoder_image_filenames = []
    segmentation_image_filenames = []

    for image_path in source_dir.glob("positive/images/*.jpg"):
        autoencoder_image_filenames.append(image_path)
        segmentation_image_filenames.append(image_path)

    for image_path in source_dir.glob("negative/images/*.jpg"):
        autoencoder_image_filenames.append(image_path)

    # Random shuffle the list of image file names
    random.shuffle(autoencoder_image_filenames)
    random.shuffle(segmentation_image_filenames)

    # Split the list of image file names into train and test sets
    autoencoder_train = autoencoder_image_filenames[:int(len(autoencoder_image_filenames) * train_ratio)]
    autoencoder_test = autoencoder_image_filenames[int(len(autoencoder_image_filenames) * train_ratio):]
    segmentation_train = segmentation_image_filenames[:int(len(segmentation_image_filenames) * train_ratio)]
    segmentation_test = segmentation_image_filenames[int(len(segmentation_image_filenames) * train_ratio):]

    # Write the train and test sets to text files
    with open(train_autoencoder_file, "w") as f:
        f.write("\n".join([str(PurePosixPath(path.relative_to(source_dir))) for path in autoencoder_train]))
    with open(test_autoencoder_file, "w") as f:
        f.write("\n".join([str(PurePosixPath(path.relative_to(source_dir))) for path in autoencoder_test]))
    with open(train_segmentation_file, "w") as f:
        f.write("\n".join([str(PurePosixPath(path.relative_to(source_dir))) for path in segmentation_train]))
    with open(test_segmentation_file, "w") as f:
        f.write("\n".join([str(PurePosixPath(path.relative_to(source_dir))) for path in segmentation_test]))

    # Create directories for train and test sets
    autoencoder_train_dir = source_dir / "train_autoencoder"
    autoencoder_test_dir = source_dir / "test_autoencoder"
    segmentation_train_dir = source_dir / "train_segmentation"
    segmentation_test_dir = source_dir / "test_segmentation"

    # Copy images to train and test directories
    for image_path in tqdm(autoencoder_train):
        class_name = image_path.parent.parent.name
        (autoencoder_train_dir / class_name).mkdir(parents=True, exist_ok=True)
        new_image_path = autoencoder_train_dir / class_name / image_path.name
        shutil.copy(image_path, new_image_path)
    for image_path in tqdm(autoencoder_test):
        class_name = image_path.parent.parent.name
        (autoencoder_test_dir / class_name).mkdir(parents=True, exist_ok=True)
        new_image_path = autoencoder_test_dir / class_name / image_path.name
        shutil.copy(image_path, new_image_path)
    for image_path in tqdm(segmentation_train):
        class_name = image_path.parent.parent.name
        (segmentation_train_dir / class_name).mkdir(parents=True, exist_ok=True)
        new_image_path = segmentation_train_dir / class_name / image_path.name
        shutil.copy(image_path, new_image_path)
    for image_path in tqdm(segmentation_test):
        class_name = image_path.parent.parent.name
        (segmentation_test_dir / class_name).mkdir(parents=True, exist_ok=True)
        new_image_path = segmentation_test_dir / class_name / image_path.name
        shutil.copy(image_path, new_image_path)


if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    main_dir = current_dir.parent
    data_directory = main_dir / "data"

    split_data(data_directory)
