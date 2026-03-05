import base64
import io
import os

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
import torch

from ..data.api.api_helper import get_access_token, get_recent_correlated_vessels
from ..data.dataset import apply_transforms, build_eval_transforms, rotate_and_crop_by_heading
from ..models.reid_model import ReIDModel
from ..utils.faiss_index import load_index, load_metadata, search

load_dotenv()


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


def fetch_skylight_events(days=7):
    """Fetch recent Sentinel-2 vessel detection events from Skylight API.

    Returns a list of event dicts with fields needed by the frontend.
    Uses the same length >= 150m constraint as the training data pipeline.
    """
    username = os.getenv("SKYLIGHT_USERNAME")
    password = os.getenv("SKYLIGHT_PASSWORD")
    if not username or not password:
        raise RuntimeError("SKYLIGHT_USERNAME and SKYLIGHT_PASSWORD must be set in .env")

    access_token = get_access_token(username, password)
    response = get_recent_correlated_vessels(access_token, days)

    events = []
    for record in response["records"]:
        details = record.get("eventDetails", {})
        vessel_info = record.get("vessels", {}).get("vessel0", {})
        start = record.get("start", {})
        point = start.get("point", {})

        events.append({
            "event_id": record["eventId"],
            "event_type": record.get("eventType"),
            "mmsi": vessel_info.get("mmsi"),
            "vessel_name": vessel_info.get("name"),
            "vessel_type": vessel_info.get("vesselType"),
            "country_code": vessel_info.get("countryCode"),
            "image_url": details.get("imageUrl"),
            "estimated_length": details.get("estimatedLength"),
            "orientation": details.get("orientation"),
            "detection_score": details.get("score"),
            "lat": point.get("lat"),
            "lon": point.get("lon"),
            "time": start.get("time"),
        })

    return events


def download_image(image_url):
    """Download an image from a URL and return (raw_bytes, base64_string)."""
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    img_data = resp.content
    img_b64 = base64.b64encode(img_data).decode("utf-8")
    return img_data, img_b64
