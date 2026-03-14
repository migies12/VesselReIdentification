import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vessel_reid.model.dataset.dataset import DataConfig, SingleImageDataset
from vessel_reid.model.models.reid_model import ReIDModel
from vessel_reid.paths import GALLERY_CSV, RAW_IMAGES_DIR, MODEL_CHECKPOINT, FAISS_INDEX_PATH, FAISS_METADATA_PATH, MODEL_DIR
from vessel_reid.model.utils.config import load_config
from vessel_reid.model.inference.faiss_index import build_index, save_index, save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS gallery")
    parser.add_argument("--config", required=True, help="Path to gallery config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(
        csv_path=str(GALLERY_CSV),
        image_root=str(RAW_IMAGES_DIR),
        image_size=cfg["gallery"]["image_size"],
        use_length=cfg["gallery"]["use_length"],
        length_mean=cfg["gallery"]["length_mean"],
        length_std=cfg["gallery"]["length_std"],
    )
    dataset = SingleImageDataset(data_cfg)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = ReIDModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=cfg["model"]["embedding_dim"],
        use_length=cfg["model"]["use_length"],
        length_embed_dim=cfg["model"]["length_embed_dim"],
        pretrained=False,
    ).to(device)
    state = torch.load(str(MODEL_CHECKPOINT), map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_embeddings = []
    metadata = []

    with torch.no_grad():
        for images, lengths, boat_ids, image_paths, length_values in tqdm(loader, desc="gallery"):
            images = images.to(device)
            if lengths is not None:
                lengths = lengths.to(device)
            embeddings = model(images, lengths)
            all_embeddings.append(embeddings.cpu().numpy())

            for boat_id, image_path, length_value in zip(boat_ids, image_paths, length_values):
                metadata.append(
                    {
                        "boat_id": str(boat_id),
                        "image_path": image_path,
                        "length_m": None if length_value is None else float(length_value),
                    }
                )

    embeddings = np.concatenate(all_embeddings, axis=0)
    index = build_index(embeddings, normalize=cfg["faiss"]["normalize"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_index(index, str(FAISS_INDEX_PATH))
    save_metadata(metadata, str(FAISS_METADATA_PATH))

    print(f"saved index to {FAISS_INDEX_PATH}")
    print(f"saved metadata to {FAISS_METADATA_PATH}")


if __name__ == "__main__":
    main()
