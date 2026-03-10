import cv2
import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans

from .config import BG_VALUE, DILATION_KERNEL_SIZE, DILATION_ITERATIONS
from . import data_utils
from vessel_reid.paths import (
    CROPPED_IMAGES_DIR as INPUT_DIR,
    NORMALIZED_IMAGES_DOR as OUTPUT_DIR
)


def cluster(image: Image.Image, n_clusters=2):
    """
    Returns an np array mask of the boat using k-clustering
    """
    img = np.array(image)
    h, w, c = img.shape
    flat_img = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(flat_img)
    labels_img = labels.reshape(h, w)

    counts = np.bincount(labels)
    boat_label = np.argmin(counts) # Assumes the boat takes up less space in the image than the water

    mask = (labels_img == boat_label).astype(np.uint8)
    return mask

def dilate(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)

def normalize_background(image: Image.Image):
    """
    Replace background pixels with gray
    """
    mask = cluster(image)
    dilated_mask = dilate(mask, DILATION_KERNEL_SIZE, DILATION_ITERATIONS)
    img = np.array(image)
    img[dilated_mask == 0] = BG_VALUE
    return Image.fromarray(img)

if __name__ == "__main__":
    data_utils.setup_dryrun_folder(OUTPUT_DIR)

    processed_images = 0
    total_images = len(os.listdir(INPUT_DIR))
    for filename in os.listdir(INPUT_DIR):
        input_path = INPUT_DIR / filename
        image = Image.open(input_path).convert("RGB")

        normalized = normalize_background(image)

        output_path = OUTPUT_DIR / filename
        normalized.save(output_path)

        processed_images += 1
        print(f"Normalized image {processed_images} of {total_images}: {filename}")
