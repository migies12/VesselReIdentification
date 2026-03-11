"""
This file contains both a standalone script for filtering out the cloudy images from an existing dataset,
and a helper function for filtering out cloudy images when fetching them from the API (`is_cloudy_bytes`)

Both methods rely on the same criteria for classifying an image as "cloudy", contained in the `is_cloudy` function
This function uses thresholds for brightness and saturation that can be adjusted. I found their current values through
trial and error and manual inspection, and the results are moderately successful
"""

import numpy as np
import pandas as pd
import rasterio

from vessel_reid.paths import (
    RAW_IMAGES_DIR      as DATASET_PATH,
    RAW_METADATA_CSV    as MASTER_CSV_PATH,
    FILTERED_METADATA_CSV as FILTERED_CSV_PATH
)

from .config import (
    MIN_IMAGES_PER_VESSEL,
    BRIGHTNESS_THRESHOLD,
    SATURATION_THRESHOLD,
    COVERAGE_THRESHOLD,
    LUMINANCE_R,
    LUMINANCE_G,
    LUMINANCE_B,
)


def compute_cloud_coverage(data):
    """Returns the fraction of pixels identified as cloud."""
    luminance = (LUMINANCE_R * data[0]) + (LUMINANCE_G * data[1]) + (LUMINANCE_B * data[2])
    colour_diff = np.max(data, axis=0) - np.min(data, axis=0)
    cloud_mask = (luminance > BRIGHTNESS_THRESHOLD) & (colour_diff < SATURATION_THRESHOLD)
    return float(np.sum(cloud_mask) / cloud_mask.size)


########################## Standalone Script ##########################

def cloud_coverage_filepath(image_path):
    """
    Same as `is_cloudy_bytes`, but operates on the filepath of a downloaded image.
    For use in the below script, if cloud filtering occurs AFTER data fetch.
    If csv_path is provided, uses cached cloud coverage from the CSV if available.
    """
    with rasterio.open(image_path) as img:
        if img.count < 3:
            return False
        data = img.read([1, 2, 3]).astype(float)
    return compute_cloud_coverage(data)


def is_cloudy_bytes(image_bytes):
    """
    Returns True if image is too cloudy, else False.
    For integration with the data fetching script if we need it,
    because it will allow filtering before saving the image locally.
    """
    with rasterio.MemoryFile(image_bytes) as memfile:
        with memfile.open() as img:
            if img.count < 3:
                return True
            data = img.read([1, 2, 3]).astype(float)

    return compute_cloud_coverage(data)


if __name__ == "__main__":
    print("Finding non-cloudy images...")

    df = pd.read_csv(MASTER_CSV_PATH)
    total_images = len(df)

    def process_row(row):
        img_path = DATASET_PATH / row["image_path"]
        if img_path.exists():
            return cloud_coverage_filepath(img_path)
        return 1.0
    
    df["cloud_coverage"] = df.apply(process_row, axis=1)

    df = df[df["cloud_coverage"] <= COVERAGE_THRESHOLD].copy()
    cloudy_images = total_images - len(df)
    non_cloudy_images = len(df)
    print(f"Removed {cloudy_images} cloudy images of {total_images} total images")

    vessel_counts = df.groupby("boat_id")["image_path"].transform("count")
    df = df[vessel_counts >= MIN_IMAGES_PER_VESSEL].copy()
    final_count = len(df)
    not_enough_instances_count = non_cloudy_images - final_count
    print(f"Eliminated {not_enough_instances_count} images because less than {MIN_IMAGES_PER_VESSEL} images of that vessel")
    print(f"Writing {final_count} images to output csv")

    df = df.drop(columns=["cloud_coverage"])
    df.to_csv(FILTERED_CSV_PATH, index=False)
