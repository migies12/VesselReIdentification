import argparse

import numpy as np
import torch
from PIL import Image

from vessel_reid.data.dataset import apply_transforms, build_eval_transforms, rotate_and_crop_by_heading
from vessel_reid.models.reid_model import ReIDModel
from vessel_reid.utils.config import load_config
from vessel_reid.utils.faiss_index import load_index, load_metadata, search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReID inference")
    parser.add_argument("--config", required=True, help="Path to inference config YAML")
    parser.add_argument("--image", required=True, help="Path to query image")
    parser.add_argument("--length-m", type=float, default=None, help="Vessel length in meters")
    parser.add_argument("--heading", type=float, default=None, help="Vessel heading in degrees (0-360)")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReIDModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=cfg["model"]["embedding_dim"],
        use_length=cfg["model"]["use_length"],
        length_embed_dim=cfg["model"]["length_embed_dim"],
        pretrained=False,
    ).to(device)
    checkpoint_path = args.checkpoint or cfg["model"]["checkpoint"]
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    index = load_index(cfg["faiss"]["index_path"])
    metadata = load_metadata(cfg["faiss"]["metadata_path"])

    transform = build_eval_transforms(cfg["query"]["image_size"])
    image = Image.open(args.image).convert("RGB")

    rotate_by_direction = cfg["query"].get("rotate_by_direction", False)
    if rotate_by_direction and args.heading is not None:
        image = rotate_and_crop_by_heading(image, args.heading)

    image = apply_transforms(image, transform).unsqueeze(0).to(device)

    length_tensor = None
    if cfg["query"]["use_length"]:
        if args.length_m is None:
            raise ValueError("--length-m is required when use_length=true")
        length = (args.length_m - cfg["query"]["length_mean"]) / (cfg["query"]["length_std"] + 1e-6)
        length_tensor = torch.tensor([[length]], dtype=torch.float32).to(device)

    with torch.no_grad():
        embedding = model(image, length_tensor).cpu().numpy().astype(np.float32)

    distances, indices = search(
        index,
        embedding,
        top_k=cfg["query"]["top_k"],
        normalize=cfg["faiss"]["normalize"],
    )

    top_scores = distances[0].tolist()
    top_indices = indices[0].tolist()

    results = []
    for score, idx in zip(top_scores, top_indices):
        if idx < 0 or idx >= len(metadata):
            continue
        results.append({"score": float(score), **metadata[idx]})

    if results and results[0]["score"] >= cfg["query"]["similarity_threshold"]:
        print("match found")
        print(results[0])
    else:
        print("no confident match")

    print("top_k results:")
    for item in results:
        print(item)


if __name__ == "__main__":
    main()
