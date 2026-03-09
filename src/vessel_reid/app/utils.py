import base64
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import io
import numpy as np
import os
from PIL import Image
import requests
import torch

from . import db

from ..data.api_helper_skylight import get_access_token, get_event, get_recent_correlated_vessels
from ..data.dataset import apply_transforms, build_eval_transforms, rotate_and_crop_by_heading
from ..data.filter_clouds import is_cloudy_bytes
from ..models.reid_model import ReIDModel
from ..utils.faiss_index import load_index, load_metadata, search

load_dotenv()
USERNAME = os.getenv("SKYLIGHT_USERNAME")
PASSWORD = os.getenv("SKYLIGHT_PASSWORD")
if not USERNAME or not PASSWORD:
    raise RuntimeError("SKYLIGHT_USERNAME and SKYLIGHT_PASSWORD must be set in .env")
ACCESS_TOKEN = get_access_token(USERNAME, PASSWORD)

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
    """
    Fetch recent Sentinel-2 vessel detection events from Skylight API

    Returns a list of events with fields needed by the frontend
    Uses the same length >= 150m constraint as the training data pipeline
    Also filters out events with cloudy
    """
    # Create cloudy_cache table in case it doesn't already exist (if it does, this won't do anything)
    db.init_cloudy_table()

    response = get_recent_correlated_vessels(ACCESS_TOKEN, days)
    records = response["records"]

    new_events = []
    cached_events = []

    for record in records:
        status = db.get_cached_cloudy_status(record["eventId"])
        if status is False:
            cached_events.append(record)
        elif status is None:
            new_events.append(record)

    with ThreadPoolExecutor(max_workers=20) as executor:
        new_results = list(executor.map(process_event_helper, new_events))

    cached_events = [format_event_helper(event) for event in cached_events]
    new_events = [event for event in new_results if event is not None]

    return cached_events + new_events

def process_event_helper(record):
    """
    Helper function to run in parallel when fetching Skylight events

    Checks database to see if cloudiness of image is known, otherwise checks it and stores result
    """
    try:
        event_id = record["eventId"]
        cached_status = db.get_cached_cloudy_status(event_id)
        if cached_status is True:
            return None
        elif not cached_status:
            details = record.get("eventDetails", {})
            img_data, _ = download_image(details.get("imageUrl"))
            is_cloudy = is_cloudy_bytes(img_data)
            db.cache_cloudy_status(event_id, is_cloudy)
            if is_cloudy:
                return None
            
        return format_event_helper(record)
    
    except Exception as e:
        return None
    
def format_event_helper(record):
    details = record.get("eventDetails", {})
    vessel_info = record.get("vessels", {}).get("vessel0", {})
    start = record.get("start", {})
    point = start.get("point", {})

    return {
        "event_id": record["eventId"],
            "event_type": record.get("eventType"),
            "mmsi": vessel_info.get("mmsi"),
            "vessel_name": vessel_info.get("name"),
            "vessel_type": vessel_info.get("vesselType"),
            "country_code": vessel_info.get("countryCode"),
            "image_url": details.get("imageUrl"),
            "estimated_length": details.get("estimatedLength"),
            "heading": details.get("heading"),
            "detection_score": details.get("score"),
            "lat": point.get("lat"),
            "lon": point.get("lon"),
            "time": start.get("time"),
    }

def get_event_by_id(event_id):
    return get_event(ACCESS_TOKEN, event_id)

def download_image(image_url):
    """Download an image from a URL and return (raw_bytes, base64_string)."""
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    img_data = resp.content
    img_b64 = base64.b64encode(img_data).decode("utf-8")
    return img_data, img_b64
