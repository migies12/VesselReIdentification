import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vessel_reid.model.dataset.dataset import DataConfig, SingleImageDataset
from vessel_reid.model.models.reid_model import ReIDModel


def encode_dataset(
    model: ReIDModel,
    dataset_cfg: DataConfig,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """
    runs images through the model and gets embeddings
    """
    dataset = SingleImageDataset(dataset_cfg)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_embeddings = []
    all_ids = []

    model.eval()
    with torch.no_grad():
        for images, lengths, boat_ids, _image_paths, _length_values in tqdm(loader, desc="encoding"):
            images = images.to(device)
            if lengths is not None:
                lengths = lengths.to(device)
            embeddings = model(images, lengths)
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend([str(b) for b in boat_ids])

    return np.concatenate(all_embeddings, axis=0), all_ids


def build_similarity_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """
    similarity matrix of query pics vs gallery pics
    shape: num_queries x num_gallery. Higher = more similar.
    """
    return query_embeddings @ gallery_embeddings.T
