import io
import os
from PIL import Image

from . import data_utils

from vessel_reid.paths import (
    FILTERED_IMAGES_DIR as INPUT_PATH,
    CROPPED_IMAGES_DIR as OUTPUT_PATH
)

from .config import CROP_FRACTION

"""
Proof-of-concept script for cropping images
Intended as an intermediate step to make it easier to experiment with and adjust parameters
Once this is satisfactory, the functionality should be integrated into the image augmentations rather than used on its own
"""

def crop_water(image_bytes):
    """
    Crops the outer CROP_FRACTION of the image
    Returns bytes of the cropped image
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    crop_margin = int(w * CROP_FRACTION / 2)

    image = image.crop((
        crop_margin,
        crop_margin,
        w - crop_margin,
        h - crop_margin
    ))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

if __name__ == "__main__":
    data_utils.setup_dryrun_folder(OUTPUT_PATH)

    processed_images = 0
    total_images = len(os.listdir(INPUT_PATH))
    for filename in os.listdir(INPUT_PATH):
        input_path = INPUT_PATH / filename
        with open(input_path, "rb") as infile:
            image_bytes = infile.read()
            cropped_bytes = crop_water(image_bytes)
            output_path = OUTPUT_PATH / filename
            with open(output_path, "wb") as outfile:
                outfile.write(cropped_bytes)
                processed_images += 1
                print(f"Cropped image {processed_images} of {total_images}: {filename}")
