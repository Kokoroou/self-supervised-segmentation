from pathlib import Path
from typing import Union

import cv2
from tqdm import tqdm


def inspect_shape(source_loc: str, image_directory: Union[Path, str]) -> list:
    """
    Check if 2 image from 2 folder (original and mask) have the same shape

    :param source_loc: Location of hospital where the image is taken or folder name
    :param image_directory: Base directory of the image
    :return:
    """
    image_directory = Path(image_directory)

    image_dir_1 = image_directory / f"images_{source_loc}"
    image_dir_2 = image_directory / f"masks_{source_loc}"

    distinct_shapes = set()

    for image_path_1 in tqdm(list(image_dir_1.glob("*.jpg")), desc=source_loc):
        image_stem_1 = image_path_1.stem
        image_name_1 = image_path_1.name
        image_name_2 = image_stem_1 + "_mask.jpg"

        image_path_2 = image_dir_2 / image_name_2

        if not image_path_2.exists():
            print(f"{image_name_2} not exists")
            continue

        image_1 = cv2.imread(str(image_path_1))
        image_2 = cv2.imread(str(image_path_2))

        if image_1.shape != image_2.shape:
            print(f"{image_name_1} and {image_name_2} are not the same shape: {image_1.shape} != {image_2.shape}")

        distinct_shapes.add(image_1.shape)

    return sorted(distinct_shapes)


if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    main_dir = current_dir.parent
    data_dir = main_dir / "data"

    locations = ["C" + str(i) for i in range(1, 7)]
    locations += ["seq" + str(i) for i in range(1, 24)]

    for location in locations:
        if location.startswith("C"):
            image_dir = data_dir / "raw" / "PolypGen2021_MultiCenterData_v3" / f"data_{location}"
        elif location.startswith("seq"):
            image_dir = data_dir / "raw" / "PolypGen2021_MultiCenterData_v3" / "sequenceData" / "positive" / \
                        f"{location}"
        else:
            raise ValueError("Unknown source")

        shapes = inspect_shape(source_loc=location, image_directory=image_dir)

        print(f"Distinct shapes of location {location} ({len(shapes)}): {sorted(shapes)}")
