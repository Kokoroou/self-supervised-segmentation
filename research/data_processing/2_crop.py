import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

crop_map = {
    # Map from (height, width, channel) to (x1, y1, x2, y2) to crop
    "C1": {
        (498, 572, 3): (0, 0, 572, 498),
        (513, 628, 3): (0, 0, 628, 513),
        (1042, 1008, 3): (0, 0, 1008, 1042),
        (1048, 1232, 3): (0, 0, 1232, 1048),
        (1080, 1350, 3): (0, 0, 1350, 1080),
    },
    "C2": {
        (544, 672, 3): (25, 8, 655, 538),
        (1024, 1280, 3): (302, 8, 1202, 1018),
        (1056, 1432, 3): (0, 0, 1432, 1056),
        (1063, 1359, 3): (0, 0, 1359, 1063),
        (1063, 1383, 3): (0, 0, 1383, 1063),
        (1063, 1439, 3): (0, 0, 1439, 1063),
        (1064, 1360, 3): (0, 0, 1360, 1064),
        (1064, 1440, 3): (0, 0, 1440, 1064),
        (1064, 1720, 3): (360, 0, 1720, 1064),
        (1064, 1892, 3): (0, 0, 1892, 1064),
        (1072, 1704, 3): (350, 0, 1704, 1072),
        (1072, 1728, 3): (368, 0, 1728, 1072),
        (1072, 1912, 3): (449, 0, 1912, 1072),
        (1079, 1919, 3): (449, 0, 1919, 1079),
        (1080, 1920, 3): (450, 0, 1920, 1080),
    },
    "C3": {
        (576, 720, 3): (70, 20, 715, 560),
        (1040, 1240, 3): (0, 0, 1240, 1040),
        (1080, 1440, 3): (70, 0, 1420, 1080),
    },
    "C4": {
        (1080, 1920, 3): [(550, 0, 1900, 1080), (455, 0, 1805, 1080)]
    },
    "C5": {
        (288, 384, 3): (95, 25, 375, 265),
        (1080, 1920, 3): (670, 0, 1915, 1080)
    },
    "C6": {
        (768, 1024, 3): (84, 8, 1004, 768),
        (1080, 1920, 3): [(150, 10, 1910, 1080), (465, 10, 1885, 1080)]
    },
    "seq1": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq2": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq3": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq4": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq5": {
        (1064, 1440, 3): (70, 0, 1420, 1064)
    },
    "seq6": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq7": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq8": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq9": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq10": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq11": {
        (576, 720, 3): (44, 14, 674, 564)
    },
    "seq12": {
        (1024, 1280, 3): (304, 6, 1204, 1016)
    },
    "seq13": {
        (1064, 1440, 3): (70, 0, 1420, 1064)
    },
    "seq14": {
        (1064, 1440, 3): (70, 0, 1420, 1064)
    },
    "seq15": {
        (1072, 1704, 3): (350, 4, 1700, 1068)
    },
    "seq16": {
        (720, 1280, 3): (365, 5, 1275, 715)
    },
    "seq17": {
        (720, 1280, 3): (404, 10, 1234, 710)
    },
    "seq18": {
        (720, 1280, 3): (480, 12, 1160, 707)
    },
    "seq19": {
        (720, 1280, 3): (480, 12, 1160, 707)
    },
    "seq20": {
        (720, 1280, 3): (365, 5, 1275, 715)
    },
    "seq21": {
        (720, 1280, 3): (365, 5, 1275, 715)
    },
    "seq22": {
        (720, 1280, 3): (404, 10, 1234, 710)
    },
    "seq23": {
        (720, 1280, 3): (445, 0, 1275, 720)
    }
}
crop_map_negative = {
    "seq1": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq2": {
        (1080, 1920, 3): (455, 0, 1805, 1080)
    },
    "seq3": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq4": {
        (576, 720, 3): (44, 14, 674, 564)
    },
    "seq5": {
        (576, 720, 3): (44, 14, 674, 564)
    },
    "seq6": {
        (576, 720, 3): (44, 14, 674, 564)
    },
    "seq7": {
        (576, 720, 3): (44, 14, 674, 564)
    },
    "seq8": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq9": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq10": {
        (1080, 1920, 3): (550, 0, 1900, 1080)
    },
    "seq11": {
        (1080, 1920, 3): (670, 0, 1915, 1080)
    },
    "seq12": {
        (1080, 1920, 3): (670, 0, 1915, 1080)
    },
    "seq13": {
        (1080, 1920, 3): (670, 0, 1915, 1080)
    },
    "seq14": {
        (1080, 1920, 3): (605, 15, 1850, 1065)
    },
    "seq15": {
        (1080, 1920, 3): (605, 15, 1850, 1065)
    },
    "seq16": {
        (1080, 1920, 3): (605, 15, 1850, 1065)
    },
    "seq17": {
        (1080, 1920, 3): (605, 15, 1850, 1065)
    },
    "seq18": {
        (1080, 1920, 3): (605, 15, 1850, 1065)
    },
    "seq19": {
        (1080, 1920, 3): (720, 0, 1735, 1065)
    },
    "seq20": {
        (1080, 1920, 3): (605, 15, 1850, 1065)
    },
    "seq21": {
        (1080, 1920, 3): (720, 0, 1735, 1065)
    },
    "seq22": {
        (1080, 1920, 3): (605, 15, 1850, 1065)
    },
    "seq23": {
        (1080, 1920, 3): (605, 15, 1850, 1065)
    }
}


def crop_img(img, mask, location, img_stem=None):
    if location not in crop_map:
        return img, mask
    if img.shape not in crop_map[location]:
        return img, mask

    if location == "C4":
        special_cases = [str(i) + "_" for i in [6, 10, 13, 19, 20, 27, 29, 30, 33]]

        if any([img_stem.startswith(sc) for sc in special_cases]):
            crop = crop_map[location][img.shape][1]
        else:
            crop = crop_map[location][img.shape][0]
    elif location == "C6":
        if img.shape == (1080, 1920, 3):
            # Get sum intensity of a patch in location (250, 250, 250, 250)
            patch = img[250:450, 250:450]
            intensity = np.sum(patch)

            if intensity > 200000:
                crop = crop_map[location][img.shape][0]
            else:
                crop = crop_map[location][img.shape][1]
        else:
            crop = crop_map[location][img.shape]
    else:
        crop = crop_map[location][img.shape]

    if len(crop) == 4:
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    else:
        return img, mask


def crop_img_negative(img, location):
    if location not in crop_map_negative:
        return img
    if img.shape not in crop_map_negative[location]:
        return img

    crop = crop_map_negative[location][img.shape]

    if len(crop) == 4:
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2]
    else:
        return img


def crop_location(location, data_dir):
    if location not in crop_map:
        return

    if location.startswith("seq"):
        source_dir = data_dir / "raw" / "PolypGen2021_MultiCenterData_v3" / "sequenceData" / "positive" / f"{location}"
    elif location.startswith("C"):
        source_dir = data_dir / "raw" / "PolypGen2021_MultiCenterData_v3" / f"data_{location}"
    else:
        return

    target_dir = data_dir / "processed" / "PolypGen2021_MultiCenterData_v3" / "positive"

    os.makedirs(target_dir / "images", exist_ok=True)
    os.makedirs(target_dir / "masks", exist_ok=True)

    for image_path in tqdm(list(Path(source_dir, f"images_{location}").glob("*.jpg")), desc=location):
        image_name = image_path.name
        image_stem = image_path.stem

        if Path(target_dir, "images", f"{location}_{image_name}").exists():
            continue

        mask_path = source_dir / f"masks_{location}" / (image_stem + "_mask.jpg")

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))

        image, mask = crop_img(image, mask, location, img_stem=image_stem)

        cv2.imwrite(str(target_dir / "images" / f"{location}_{image_name}"), image)
        cv2.imwrite(str(target_dir / "masks" / f"{location}_{image_name}"), mask)


def crop_location_negative(location, data_dir):
    if location not in crop_map_negative:
        return

    if location.startswith("seq"):
        source_dir = data_dir / "raw" / "PolypGen2021_MultiCenterData_v3" / "sequenceData" / "negativeOnly" / \
                     f"{location}_neg"
    else:
        return

    target_dir = data_dir / "processed" / "PolypGen2021_MultiCenterData_v3" / "negative" / "images"

    os.makedirs(target_dir, exist_ok=True)

    for image_path in tqdm(list(source_dir.glob("*.jpg")), desc=location):
        image_name = image_path.name

        if Path(target_dir, f"{location}_{image_name}").exists():
            continue

        image = cv2.imread(str(image_path))

        image = crop_img_negative(image, location)

        cv2.imwrite(str(target_dir / f"{location}_{image_name}"), image)


if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    main_dir = current_dir.parent
    data_directory = main_dir / "data"

    locs = [f"C{i}" for i in range(1, 7)]
    locs += [f"seq{i}" for i in range(1, 24)]
    for loc in locs:
        crop_location(location=loc, data_dir=data_directory)

    locs = [f"seq{i}" for i in range(1, 24)]
    for loc in locs:
        crop_location_negative(location=loc, data_dir=data_directory)
