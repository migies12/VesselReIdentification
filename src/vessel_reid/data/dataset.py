from collections import defaultdict
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, Sampler


@dataclass
class DataConfig:
    csv_path: str
    image_root: str
    image_size: int
    use_length: bool
    length_mean: float
    length_std: float
    rotate_by_direction: bool = False
    augment: bool = False


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


def build_train_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.03),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=10, p=0.25),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_transforms(image_size: int) -> A.Compose:
    return build_eval_transforms(image_size)


def apply_transforms(image: Image.Image, transform: A.Compose) -> torch.Tensor:
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return transformed["image"]


class TripletDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv_path)
        self.transform = build_train_transforms(cfg.image_size) if cfg.augment else build_eval_transforms(cfg.image_size)
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

        image = apply_transforms(image, self.transform)
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
        self.transform = build_eval_transforms(cfg.image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = f"{self.cfg.image_root}/{row['image_path']}"
        image = Image.open(image_path).convert("RGB")

        if self.cfg.rotate_by_direction and pd.notna(row.get("heading")):
            heading = float(row["heading"])
            image = rotate_and_crop_by_heading(image, heading)

        image = apply_transforms(image, self.transform)
        length_tensor = torch.zeros(1, dtype=torch.float32)
        length_value = None
        if self.cfg.use_length:
            length = float(row["length_m"])
            length_value = length
            length = (length - self.cfg.length_mean) / (self.cfg.length_std + 1e-6)
            length_tensor = torch.tensor([length], dtype=torch.float32)
        return image, length_tensor, row["boat_id"], row["image_path"], length_value


class LabeledImageDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv_path)
        self.transform = build_train_transforms(cfg.image_size) if cfg.augment else build_eval_transforms(cfg.image_size)
        self.indices_by_id: Dict[str, List[int]] = defaultdict(list)
        for idx, boat_id in enumerate(self.df["boat_id"].astype(str).tolist()):
            self.indices_by_id[boat_id].append(idx)
        self.boat_ids = list(self.indices_by_id.keys())
        self.id_to_index = {boat_id: i for i, boat_id in enumerate(self.boat_ids)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = f"{self.cfg.image_root}/{row['image_path']}"
        image = Image.open(image_path).convert("RGB")

        if self.cfg.rotate_by_direction and pd.notna(row.get("heading")):
            heading = float(row["heading"])
            image = rotate_and_crop_by_heading(image, heading)

        image = apply_transforms(image, self.transform)
        length_tensor = torch.zeros(1, dtype=torch.float32)
        if self.cfg.use_length:
            length = float(row["length_m"])
            length = (length - self.cfg.length_mean) / (self.cfg.length_std + 1e-6)
            length_tensor = torch.tensor([length], dtype=torch.float32)
        boat_id = str(row["boat_id"])
        return image, length_tensor, self.id_to_index[boat_id]


class PKBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        indices_by_id: Dict[str, List[int]],
        p: int,
        k: int,
        batches_per_epoch: Optional[int] = None,
        seed: int = 1337,
    ) -> None:
        self.indices_by_id = indices_by_id
        self.p = p
        self.k = k
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.boat_ids = list(indices_by_id.keys())
        self._epoch = 0

    def __len__(self) -> int:
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch
        return max(len(self.boat_ids) // self.p, 1)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        boat_ids = self.boat_ids[:]
        rng.shuffle(boat_ids)

        batches_target = self.__len__()
        batch_count = 0
        idx = 0
        while batch_count < batches_target:
            if idx + self.p > len(boat_ids):
                rng.shuffle(boat_ids)
                idx = 0
            selected_ids = boat_ids[idx : idx + self.p]
            idx += self.p

            batch = []
            for boat_id in selected_ids:
                pool = self.indices_by_id[boat_id]
                if len(pool) >= self.k:
                    batch.extend(rng.sample(pool, self.k))
                else:
                    batch.extend(rng.choices(pool, k=self.k))

            rng.shuffle(batch)
            yield batch
            batch_count += 1
