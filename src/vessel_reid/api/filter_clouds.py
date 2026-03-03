import numpy as np
import rasterio
import os
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "../../../data/images"
DRY_RUN = True

def is_cloudy(image_path, brightness_threshold=210, coverage_threshold=0.15):
    """
    Returns True if image is cloudy, else False
    """
    with rasterio.open(image_path) as img:
        if img.count < 3:
            return False
        data = img.read([1, 2, 3])

    cloud_mask = np.all(data > brightness_threshold, axis=0)
    cloud_fraction = np.sum(cloud_mask) / cloud_mask.size
    return cloud_fraction > coverage_threshold

if __name__ == "__main__":
    total_images = 0
    cloudy_images = 0

    for filename in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, filename)
        if is_cloudy(path):
            print(f"Removing {filename}: Too cloudy")
            cloudy_images += 1
            if not DRY_RUN:
                os.remove(path)
        total_images += 1

    print(f"Removed {cloudy_images} cloudy images from {total_images} total images")
    print(f"{total_images - cloudy_images} images remaining")