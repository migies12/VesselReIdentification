import csv
import math
from pathlib import Path
from PIL import Image
import os

from vessel_reid.data import data_utils
from vessel_reid.paths import (
    FILTERED_IMAGES_DIR as INPUT_DIR,
    ROTATED_IMAGES_DIR as OUTPUT_DIR,
    RAW_METADATA_CSV
)

def rotate(image: Image.Image, heading: float) -> Image.Image:
    """
    Rotate image to normalize vessel heading and crop to the largest inscribed square.

    Args:
        image: PIL Image to transform
        heading: Vessel heading in degrees (0-360, where 0 is north)

    Returns:
        Rotated and cropped PIL Image
    """
    w, h = image.size

    # Rotate image by negative heading to normalize all vessels to face north (0 degrees)
    # expand=True keeps the full rotated image without clipping corners
    rotated = image.rotate(heading, resample=Image.BILINEAR, expand=True)

    centre_x = rotated.width // 2
    centre_y = rotated.height // 2
    cropped = rotated.crop((
        centre_x - (w // 2),
        centre_y - (h // 2),
        centre_x + (w // 2),
        centre_y + (h // 2)
    ))

    return cropped

if __name__ == "__main__":
    data_utils.setup_dryrun_folder(OUTPUT_DIR)

    headings = {}
    with open(RAW_METADATA_CSV, newline="") as csvfile:
        print(f"Reading headings from {RAW_METADATA_CSV}")
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                headings[Path(row["image_path"]).name] = float(row["heading"])
            except Exception as e:
                continue

    print(headings["710000200_S2C_MSIL1C_20260307T124251_N0512_R009_T25LBK_20260307T155859.SAFE_5.png"])
    
    processed_images = 0
    total_images = len(os.listdir(INPUT_DIR))
    for filename in os.listdir(INPUT_DIR):
        heading = headings.get(filename)
        if heading is None:
            print(f"Warning: no heading found for file {filename}")
            continue

        image_path = INPUT_DIR / filename
        image = Image.open(image_path).convert("RGB")
        rotated = rotate(image, heading)

        output_path = OUTPUT_DIR / filename
        rotated.save(output_path)

        processed_images += 1
        print(f"Processed {processed_images} of {total_images} images: {filename}")

