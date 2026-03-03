import numpy as np
import rasterio
import os
from pathlib import Path
import shutil

DATASET_PATH = Path(__file__).resolve().parent / "../../../data/images"
OUTPUT_PATH = Path(__file__).resolve().parent / "dryrun_filtered_images"
FILTERED_PATH = Path(__file__).resolve().parent / "dryrun_deleted_images"

DRY_RUN = True

BRIGHTNESS_THRESHOLD = 115
SATURATION_THRESHOLD = 65
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

    luminance = (LUMINANCE_R * data[0]) + (LUMINANCE_G * data[1]) + (LUMINANCE_B * data[2])
    colour_diff = np.max(data, axis=0) - np.min(data, axis=0)
    cloud_mask = (luminance > BRIGHTNESS_THRESHOLD) & (colour_diff < SATURATION_THRESHOLD)
    cloud_fraction = np.sum(cloud_mask) / cloud_mask.size

    return cloud_fraction > COVERAGE_THRESHOLD

def setup_dryrun_folder(path):
    if path.exists():
        print(f"Clearing existing files in {OUTPUT_PATH}")
        shutil.rmtree(path, ignore_errors=True)
    
    path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    total_images = 0
    cloudy_images = 0

    setup_dryrun_folder(OUTPUT_PATH)
    setup_dryrun_folder(FILTERED_PATH)

    for filename in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, filename)
        if is_cloudy(path):
            print(f"Removing {filename}: Too cloudy")
            cloudy_images += 1
            if not DRY_RUN:
                os.remove(path)
            else:
                shutil.copy2(path, FILTERED_PATH / filename)
        else:
            if DRY_RUN:
                shutil.copy2(path, OUTPUT_PATH / filename)
        total_images += 1

    print(f"Removed {cloudy_images} cloudy images from {total_images} total images")
    print(f"{total_images - cloudy_images} images remaining")