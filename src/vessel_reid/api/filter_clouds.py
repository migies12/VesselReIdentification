import numpy as np
import rasterio
import os
from pathlib import Path
import shutil

DATASET_PATH = Path(__file__).resolve().parent / "../../../data/images"
OUTPUT_PATH = Path(__file__).resolve().parent / "dryrun_filtered_images"

DRY_RUN = True

BRIGHTNESS_THRESHOLD = 150
COVERAGE_THRESHOLD = 0.05

LUMINANCE_R = 0.3
LUMINANCE_G = 0.6
LUMINANCE_B = 0.1

def is_cloudy(image_path):
    """
    Returns True if image is too cloudy, else False
    """
    with rasterio.open(image_path) as img:
        if img.count < 3:
            return False
        data = img.read([1, 2, 3])

    # Luminance
    luminance = (LUMINANCE_R * data[0]) + (LUMINANCE_G * data[1]) + (LUMINANCE_B * data[2])
    brightness_mask = luminance > BRIGHTNESS_THRESHOLD
    cloud_fraction = np.sum(brightness_mask) / brightness_mask.size

    return cloud_fraction > COVERAGE_THRESHOLD

def setup_dryrun_folder(path):
    if path.exists():
        print(f"Clearing existing files in {OUTPUT_PATH}")
        shutil.rmtree(path, ignore_errors=True)
    
    path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    total_images_count = 0
    cloudy_images_count = 0

    kept_images = []
    cloudy_images = []

    setup_dryrun_folder(OUTPUT_PATH)

    for filename in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, filename)
        if is_cloudy(path):
            print(f"Removing {filename}: Too cloudy")
            cloudy_images.append(filename)
            cloudy_images_count += 1
            if not DRY_RUN:
                os.remove(path)
        else:
            if DRY_RUN:
                shutil.copy2(path, OUTPUT_PATH / filename)
            kept_images.append(filename)
        total_images_count += 1

    with open("to_discard.txt", "w") as f:
        f.write("\n".join(cloudy_images))

    with open("to_keep.txt", "w") as f:
        f.write("\n".join(kept_images))

    print(f"Removed {cloudy_images_count} cloudy images from {total_images_count} total images")
    print(f"{total_images_count - cloudy_images_count} images remaining")