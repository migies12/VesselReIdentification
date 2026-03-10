"""
Proof-of-concept script for cropping images
Intended as an intermediate step to make it easier to experiment with and adjust parameters
Once this is satisfactory, the functionality should be integrated into the image augmentations rather than used on its own
"""
import os
from PIL import Image

from . import data_utils
from .config import CROP_FRACTION
from vessel_reid.paths import (
    ROTATED_IMAGES_DIR as INPUT_DIR,
    CROPPED_IMAGES_DIR as OUTPUT_DIR
)


def crop(image: Image.Image) -> Image.Image:
    """
    Crops the outer CROP_FRACTION of the image
    Returns bytes of the cropped image
    """
    w, h = image.size
    crop_margin = int(min(w, h) * CROP_FRACTION / 2)

    cropped = image.crop((
        crop_margin,
        crop_margin,
        w - crop_margin,
        h - crop_margin
    ))
    return cropped

if __name__ == "__main__":
    data_utils.setup_dryrun_folder(OUTPUT_DIR)

    processed_images = 0
    total_images = len(os.listdir(INPUT_DIR))
    for filename in os.listdir(INPUT_DIR):
        input_path = INPUT_DIR / filename
        image = Image.open(input_path).convert("RGB")

        cropped = crop(image)

        output_path = OUTPUT_DIR / filename
        cropped.save(output_path)

        processed_images += 1
        print(f"Cropped image {processed_images} of {total_images}: {filename}")
