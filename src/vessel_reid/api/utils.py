import io
import numpy as np
from PIL import Image
import torch

from ..data.dataset import apply_transforms, build_eval_transforms, rotate_and_crop_by_heading
from ..models.reid_model import ReIDModel
from ..utils.faiss_index import load_index, load_metadata, search

def load_model(cfg, device, model_path):
    model = ReIDModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=cfg["model"]["embedding_dim"],
        use_length=cfg["model"]["use_length"],
        length_embed_dim=cfg["model"]["length_embed_dim"],
        pretrained=False,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def transform_image(cfg, img, heading, device):
    transform = build_eval_transforms(cfg["query"]["image_size"])
    image = Image.open(io.BytesIO(img)).convert("RGB")
    rotate_by_direction = cfg["query"].get("rotate_by_direction", False)
    if rotate_by_direction and heading is not None:
        image = rotate_and_crop_by_heading(image, heading)
    image = apply_transforms(image, transform).unsqueeze(0).to(device)
    return image

def generate_embedding(cfg, image, length_m, model, device):
    length_tensor = None
    if cfg["query"]["use_length"]:
        if length_m is None:
            raise ValueError("length_m is required when use_length=true in config")
        normalized_length = (length_m - cfg["query"]["length_mean"]) / (cfg["query"]["length_std"] + 1e-6)
        length_tensor = torch.tensor([[normalized_length]], dtype=torch.float32).to(device)

    with torch.no_grad():
        embedding = model(image, length_tensor).cpu().numpy().astype(np.float32)

    return embedding

def similarity_search(cfg, embedding):
    index = load_index(cfg["faiss"]["index_path"])
    metadata = load_metadata(cfg["faiss"]["metadata_path"])

    distances, indices = search(
        index,
        embedding,
        top_k=cfg["query"]["top_k"],
        normalize=cfg["faiss"]["normalize"]
    )

    top_scores = distances[0].tolist()
    top_indices = indices[0].tolist()

    results = []
    for score, idx in zip(top_scores, top_indices):
        if idx < 0 or idx >= len(metadata):
            continue
        results.append({"score": float(score), **metadata[idx]})
    
    matched = results and results[0]["score"] >= cfg["query"]["similarity_threshold"]
    return {
        "matched": matched,
        "top_match": results[0] if results else None,
        "all_results": results,
    }
