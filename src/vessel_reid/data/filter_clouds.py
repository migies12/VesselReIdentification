"""
This file contains both a standalone script for filtering out the cloudy images from an existing dataset,
and a helper function for filtering out cloudy images when fetching them from the API (`is_cloudy_bytes`)

Both methods rely on the same criteria for classifying an image as "cloudy", contained in the `is_cloudy` function
This function uses thresholds for brightness and saturation that can be adjusted. I found their current values through
trial and error and manual inspection, and the results are moderately successful
"""

from collections import defaultdict
import os
from pathlib import Path
import shutil

import numpy as np
import rasterio

import data_utils

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
MASTER_CSV_PATH = DATASET_PATH.parent / "all_labels.csv"
OUTPUT_PATH = DATASET_PATH.parent / "dryrun_filtered_images"
FILTERED_PATH = DATASET_PATH.parent / "dryrun_deleted_images"
EXCLUDED_PATH = DATASET_PATH.parent / "dryrun_excluded_vessels"

def compute_cloud_coverage(data):
    """Returns the fraction of pixels identified as cloud."""
    luminance = (LUMINANCE_R * data[0]) + (LUMINANCE_G * data[1]) + (LUMINANCE_B * data[2])
    colour_diff = np.max(data, axis=0) - np.min(data, axis=0)
    cloud_mask = (luminance > BRIGHTNESS_THRESHOLD) & (colour_diff < SATURATION_THRESHOLD)
    return float(np.sum(cloud_mask) / cloud_mask.size)


def is_cloudy(data):
    return compute_cloud_coverage(data) > COVERAGE_THRESHOLD


def get_cloud_coverage(image_path, csv_path, rows=None):
    """
    Returns the cloud coverage fraction for the image at image_path.
    Checks csv_path for a cached value first; computes and caches if not found.
    Optionally accepts an already-loaded rows dict to avoid re-reading the CSV.
    """
    filename = Path(image_path).name
    if rows is None:
        rows = data_utils.load_csv(csv_path)

    cached = rows.get(filename, {}).get("cloud_coverage", "")
    if cached != "":
        return float(cached)

    with rasterio.open(image_path) as img:
        if img.count < 3:
            return 0.0
        data = img.read([1, 2, 3])

    coverage = compute_cloud_coverage(data)
    row = rows.get(filename, {"image_path": filename, "boat_id": "", "length_m": "", "heading": ""})
    data_utils.upsert_row(csv_path, {**row, "cloud_coverage": coverage})

    return coverage


########################## Standalone Script ##########################

def is_cloudy_filepath(image_path, csv_path=None):
    """
    Same as `is_cloudy_bytes`, but operates on the filepath of a downloaded image.
    For use in the below script, if cloud filtering occurs AFTER data fetch.
    If csv_path is provided, uses cached cloud coverage from the CSV if available.
    """
    if csv_path is not None:
        return get_cloud_coverage(image_path, csv_path) > COVERAGE_THRESHOLD
    with rasterio.open(image_path) as img:
        if img.count < 3:
            return False
        data = img.read([1, 2, 3]).astype(float)
    return is_cloudy(data)


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

    return is_cloudy(data)


def setup_dryrun_folder(path):
    """
    Helper for the script below
    """
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

    # Load CSV cache once for the whole run
    rows = data_utils.load_csv(MASTER_CSV_PATH)

    # Pass 1: group images by vessel and classify each as cloudy or clean
    vessel_clean = defaultdict(list)   # mmsi -> [filename, ...]
    vessel_cloudy = defaultdict(list)  # mmsi -> [filename, ...]

    for filename in os.listdir(DATASET_PATH):
        mmsi = filename.split("_")[0]
        path = os.path.join(DATASET_PATH, filename)
        coverage = get_cloud_coverage(path, MASTER_CSV_PATH, rows=rows)
        rows.setdefault(filename, {"image_path": filename, "boat_id": mmsi, "length_m": "", "heading": ""})
        rows[filename]["cloud_coverage"] = coverage
        if coverage > COVERAGE_THRESHOLD:
            vessel_cloudy[mmsi].append(filename)
        else:
            vessel_clean[mmsi].append(filename)
        total_images += 1

    # Write all cached coverages to CSV in one pass
    data_utils.write_csv(MASTER_CSV_PATH, rows)

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