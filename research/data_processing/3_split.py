import random
from pathlib import Path, PosixPath, PurePosixPath


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
        autoencoder_image_filenames.append(str(PurePosixPath(image_path.relative_to(source_dir))))
        segmentation_image_filenames.append(str(PurePosixPath(image_path.relative_to(source_dir))))

    for image_path in source_dir.glob("negative/images/*.jpg"):
        autoencoder_image_filenames.append(str(PurePosixPath(image_path.relative_to(source_dir))))

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
        f.write("\n".join(autoencoder_train))
    with open(test_autoencoder_file, "w") as f:
        f.write("\n".join(autoencoder_test))
    with open(train_segmentation_file, "w") as f:
        f.write("\n".join(segmentation_train))
    with open(test_segmentation_file, "w") as f:
        f.write("\n".join(segmentation_test))


if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    main_dir = current_dir.parent
    data_directory = main_dir / "data"

    split_data(data_directory)
