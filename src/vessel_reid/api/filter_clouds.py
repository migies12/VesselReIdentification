import numpy as np
import rasterio
import os
from pathlib import Path
from collections import defaultdict
import shutil

from config import (
    MIN_IMAGES_PER_VESSEL,
    DRY_RUN,
    BRIGHTNESS_THRESHOLD,
    SATURATION_THRESHOLD,
    COVERAGE_THRESHOLD,
    LUMINANCE_R,
    LUMINANCE_G,
    LUMINANCE_B,
)

DATASET_PATH = Path(__file__).resolve().parent / "../../../data/images"
OUTPUT_PATH = Path(__file__).resolve().parent / "dryrun_filtered_images"
FILTERED_PATH = Path(__file__).resolve().parent / "dryrun_deleted_images"
EXCLUDED_PATH = Path(__file__).resolve().parent / "dryrun_excluded_vessels"

def is_cloudy(data):
    luminance = (LUMINANCE_R * data[0]) + (LUMINANCE_G * data[1]) + (LUMINANCE_B * data[2])
    colour_diff = np.max(data, axis=0) - np.min(data, axis=0)
    cloud_mask = (luminance > BRIGHTNESS_THRESHOLD) & (colour_diff < SATURATION_THRESHOLD)
    cloud_fraction = np.sum(cloud_mask) / cloud_mask.size

    return cloud_fraction > COVERAGE_THRESHOLD

def is_cloudy_filepath(image_path):
    """
    Returns True if image at the filepath is too cloudy, else False
    """
    with rasterio.open(image_path) as img:
        if img.count < 3:
            return False
        data = img.read([1, 2, 3])

    return is_cloudy(data)

def is_cloudy_bytes(image_bytes):
    """
    Same as above, but takes image bytes instead of filename
    For integration with the data fetching script
    """
    with rasterio.MemoryFile(image_bytes) as memfile:
        with memfile.open() as img:
            if img.count < 3:
                return True
            data = img.read([1, 2, 3]).astype(float)

    return is_cloudy(data)

def setup_dryrun_folder(path):
    if path.exists():
        print(f"Clearing existing files in {OUTPUT_PATH}")
        shutil.rmtree(path, ignore_errors=True)
    
    path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    total_images = 0
    cloudy_images = 0
    excluded_vessels = 0
    excluded_images = 0

    setup_dryrun_folder(OUTPUT_PATH)
    setup_dryrun_folder(FILTERED_PATH)
    setup_dryrun_folder(EXCLUDED_PATH)

    # Pass 1: group images by vessel and classify each as cloudy or clean
    vessel_clean = defaultdict(list)   # mmsi -> [filename, ...]
    vessel_cloudy = defaultdict(list)  # mmsi -> [filename, ...]

    for filename in os.listdir(DATASET_PATH):
        mmsi = filename.split("_")[0]
        path = os.path.join(DATASET_PATH, filename)
        if is_cloudy_filepath(path):
            vessel_cloudy[mmsi].append(filename)
        else:
            vessel_clean[mmsi].append(filename)
        total_images += 1

    # Pass 2: enforce minimum threshold, then copy/delete
    for mmsi in set(vessel_clean) | set(vessel_cloudy):
        clean = vessel_clean[mmsi]
        cloudy = vessel_cloudy[mmsi]

        if len(clean) < MIN_IMAGES_PER_VESSEL:
            # Too few clean images remain after cloud filtering — exclude entire vessel
            print(f"Excluding vessel {mmsi}: only {len(clean)} clean images after cloud filtering (need {MIN_IMAGES_PER_VESSEL})")
            excluded_vessels += 1
            excluded_images += len(clean) + len(cloudy)
            for filename in clean + cloudy:
                path = os.path.join(DATASET_PATH, filename)
                if not DRY_RUN:
                    os.remove(path)
                else:
                    shutil.copy2(path, EXCLUDED_PATH / filename)
        else:
            for filename in cloudy:
                path = os.path.join(DATASET_PATH, filename)
                print(f"Removing {filename}: Too cloudy")
                cloudy_images += 1
                if not DRY_RUN:
                    os.remove(path)
                else:
                    shutil.copy2(path, FILTERED_PATH / filename)
            for filename in clean:
                path = os.path.join(DATASET_PATH, filename)
                if DRY_RUN:
                    shutil.copy2(path, OUTPUT_PATH / filename)

    print(f"Removed {cloudy_images} cloudy images from {total_images} total images")
    print(f"Excluded {excluded_vessels} vessels ({excluded_images} images) that fell below {MIN_IMAGES_PER_VESSEL}-image threshold after cloud filtering")
    print(f"{total_images - cloudy_images - excluded_images} images remaining")