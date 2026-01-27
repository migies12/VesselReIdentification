from collections import defaultdict
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class DataConfig:
    csv_path: str
    image_root: str
    image_size: int
    use_length: bool
    length_mean: float
    length_std: float
    rotate_by_direction: bool = False


def rotate_and_crop_by_heading(image: Image.Image, heading: float) -> Image.Image:
    """
    Rotate image to normalize vessel heading and crop to the largest inscribed square.

    Args:
        image: PIL Image to transform
        heading: Vessel heading in degrees (0-360, where 0 is north)

    Returns:
        Rotated and cropped PIL Image
    """
    # Rotate image by negative heading to normalize all vessels to face north (0 degrees)
    # expand=True keeps the full rotated image without clipping corners
    rotated = image.rotate(-heading, resample=Image.BILINEAR, expand=True)

    # Calculate the crop size: side / sqrt(2) ensures valid crop for any rotation
    # Use the original image dimensions for the calculation
    original_size = min(image.width, image.height)
    crop_size = int(original_size / math.sqrt(2))

    # Center crop the rotated image
    center_x = rotated.width // 2
    center_y = rotated.height // 2
    half_crop = crop_size // 2

    left = center_x - half_crop
    top = center_y - half_crop
    right = left + crop_size
    bottom = top + crop_size

    cropped = rotated.crop((left, top, right, bottom))
    return cropped


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TripletDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv_path)
        self.transform = build_transforms(cfg.image_size)
        self.indices_by_id: Dict[str, List[int]] = defaultdict(list)
        for idx, boat_id in enumerate(self.df["boat_id"].astype(str).tolist()):
            self.indices_by_id[boat_id].append(idx)
        self.boat_ids = list(self.indices_by_id.keys())

    def __len__(self) -> int:
        return len(self.df)

    def _load_item(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        row = self.df.iloc[idx]
        image_path = f"{self.cfg.image_root}/{row['image_path']}"
        image = Image.open(image_path).convert("RGB")

        if self.cfg.rotate_by_direction and pd.notna(row.get("heading")):
            heading = float(row["heading"])
            image = rotate_and_crop_by_heading(image, heading)

        image = self.transform(image)
        length_tensor = torch.zeros(1, dtype=torch.float32)
        if self.cfg.use_length:
            length = float(row["length_m"])
            length = (length - self.cfg.length_mean) / (self.cfg.length_std + 1e-6)
            length_tensor = torch.tensor([length], dtype=torch.float32)
        return image, length_tensor

    def __getitem__(self, idx: int):
        anchor_img, anchor_len = self._load_item(idx)
        anchor_id = str(self.df.iloc[idx]["boat_id"])

        pos_idx = idx
        same_pool = self.indices_by_id[anchor_id]
        if len(same_pool) > 1:
            while pos_idx == idx:
                pos_idx = random.choice(same_pool)
        pos_img, pos_len = self._load_item(pos_idx)

        neg_id = anchor_id
        while neg_id == anchor_id:
            neg_id = random.choice(self.boat_ids)
        neg_idx = random.choice(self.indices_by_id[neg_id])
        neg_img, neg_len = self._load_item(neg_idx)

        return (anchor_img, anchor_len), (pos_img, pos_len), (neg_img, neg_len)


class SingleImageDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv_path)
        self.transform = build_transforms(cfg.image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = f"{self.cfg.image_root}/{row['image_path']}"
        image = Image.open(image_path).convert("RGB")

        if self.cfg.rotate_by_direction and pd.notna(row.get("heading")):
            heading = float(row["heading"])
            image = rotate_and_crop_by_heading(image, heading)

        image = self.transform(image)
        length_tensor = torch.zeros(1, dtype=torch.float32)
        length_value = None
        if self.cfg.use_length:
            length = float(row["length_m"])
            length_value = length
            length = (length - self.cfg.length_mean) / (self.cfg.length_std + 1e-6)
            length_tensor = torch.tensor([length], dtype=torch.float32)
        return image, length_tensor, row["boat_id"], row["image_path"], length_value
