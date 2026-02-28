import base64
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import torch

from ..utils.config import load_config
from . import utils

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist", "assets"),
    static_url_path="/assets",
)
CORS(app)

# Global Initialization
# Load the config
api_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(api_dir, "..", "..", ".."))
cfg = load_config(os.path.join(root, "configs", "inference.yaml"))

# Resolve relative faiss paths against the project root
cfg["faiss"]["index_path"] = os.path.join(root, cfg["faiss"]["index_path"])
cfg["faiss"]["metadata_path"] = os.path.join(root, cfg["faiss"]["metadata_path"])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = utils.load_model(cfg, device, os.path.join(os.path.dirname(__file__), "model.pt"))

# In-memory event cache
_events_cache = {}


@app.route("/")
def home():
    # Serve React app if built, otherwise show API status
    index_path = os.path.join(root, "frontend", "dist", "index.html")
    if os.path.exists(index_path):
        return send_from_directory(os.path.join(root, "frontend", "dist"), "index.html")
    return "<h2>Vessel Reidentification API</h2>"


@app.route("/infer", methods=["POST"])
def infer():
    if not request.is_json:
        return jsonify({"error": "Payload must be JSON"}), 400

    data = request.get_json()
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "No 'image' key in JSON payload"}), 400

    try:
        # Apply pre-processing transformation to image
        if ";base64," in img_b64:
            img_b64 = img_b64.split(";base64,")[-1]
        img_data = base64.b64decode(img_b64)

        heading = data.get("heading")
        length = data.get("length")
        transformed_img = utils.transform_image(cfg, img_data, heading, device)

        # Generate embedding
        embedding = utils.generate_embedding(cfg, transformed_img, length, model, device)

        # Similarity search
        top_k_results = utils.similarity_search(cfg, embedding)
        return jsonify(top_k_results), 200

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/events", methods=["GET"])
def get_events():
    """Fetch recent Sentinel-2 vessel detection events from the Skylight API."""
    global _events_cache
    try:
        force_refresh = request.args.get("refresh", "").lower() == "true"

        if _events_cache and not force_refresh:
            return jsonify(list(_events_cache.values())), 200

        events = utils.fetch_skylight_events(days=7)

        _events_cache = {}
        for event in events:
            _events_cache[event["event_id"]] = event

        return jsonify(events), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch events: {str(e)}"}), 500


@app.route("/gallery-image/<path:image_path>")
def gallery_image(image_path):
    """Serve a gallery image from the dataset images directory."""
    image_root = os.path.join(root, "dataset", "images")
    return send_from_directory(image_root, image_path)


@app.route("/events/<event_id>/infer", methods=["POST"])
def infer_event(event_id):
    """Download the image for a cached event and run inference on it."""
    if event_id not in _events_cache:
        return jsonify({"error": "Event not found. Fetch /events first."}), 404

    try:
        event = _events_cache[event_id]
        img_data, img_b64 = utils.download_image(event["image_url"])

        heading = event.get("orientation")
        length = event.get("estimated_length")

        transformed_img = utils.transform_image(cfg, img_data, heading, device)
        embedding = utils.generate_embedding(cfg, transformed_img, length, model, device)
        top_k_results = utils.similarity_search(cfg, embedding)

        return jsonify({
            "event": event,
            "query_image": img_b64,
            **top_k_results,
        }), 200

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
