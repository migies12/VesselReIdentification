from collections import defaultdict
import random
from dataclasses import dataclass
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
        image = self.transform(image)
        length_tensor = torch.zeros(1, dtype=torch.float32)
        length_value = None
        if self.cfg.use_length:
            length = float(row["length_m"])
            length_value = length
            length = (length - self.cfg.length_mean) / (self.cfg.length_std + 1e-6)
            length_tensor = torch.tensor([length], dtype=torch.float32)
        return image, length_tensor, row["boat_id"], row["image_path"], length_value
