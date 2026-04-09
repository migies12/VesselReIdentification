"""
TO RUN DEVELOPMENT SERVER: 

From the `src/vessel_reid` folder:
```
export FLASK_APP=app.api
flask run --port 5001
```
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import torch

from ..paths import FAISS_INDEX_PATH, FAISS_METADATA_PATH, RAW_IMAGES_DIR
from ..utils.config import load_config
from . import db, utils

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "assets"),
    static_url_path="/assets",
)
CORS(app)

# Global Initialization
# Load the config
app_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(app_dir, ".."))
project_root = os.path.abspath(os.path.join(src_root, "..", ".."))
cfg = load_config(os.path.join(project_root, "configs", "inference.yaml"))

cfg["faiss"]["index_path"] = str(FAISS_INDEX_PATH)
cfg["faiss"]["metadata_path"] = str(FAISS_METADATA_PATH)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = utils.load_model(cfg, device, os.path.join(os.path.dirname(__file__), "model.pt"))

# In-memory event cache
_events_cache = {}

@app.route("/api/events", methods=["GET"])
def get_events():
    """Fetch recent Sentinel-2 vessel detection events from the Skylight API."""
    global _events_cache
    try:
        force_refresh = request.args.get("refresh", "").lower() == "true"

        if _events_cache and not force_refresh:
            return jsonify(list(_events_cache.values())), 200

        events = utils.fetch_skylight_events(days=10)

        _events_cache = {}
        for event in events:
            _events_cache[event["event_id"]] = event

        return jsonify(events), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch events: {str(e)}"}), 500
    
@app.route("/api/events/demo", methods=["GET"])
def get_demo_events():
    global _events_cache
    event_ids = db.get_demo_events()
    events = []
    for eid in event_ids:
        full_event = utils.get_event_by_id(eid)
        normalized = utils.format_event_helper(full_event)
        
        _events_cache[eid] = normalized
        events.append(normalized)
        
    return jsonify(events), 200

@app.route("/api/events/demo/add/<event_id>", methods=["POST"])
def add_demo_event(event_id):
    """
    Add a demo event by ID to the stored list of demo events
    """
    db.add_demo_event(event_id)
    return {"status": "success", "event_id": event_id}, 201

@app.route("/api/events/demo/remove/<path:event_id>", methods=["DELETE"])
def remove_demo_event(event_id):
    db.remove_demo_event(event_id)
    return {"status": "success"}, 200

@app.route("/api/events/<path:event_id>/", methods=["GET"])
def event(event_id):
    """
    Fetch a specific event from Skylight by event id
    """
    return utils.get_event_by_id(event_id)

@app.route("/api/gallery-image/<path:image_path>")
def gallery_image(image_path):
    """Serve a gallery image from the dataset images directory."""
    image_root = str(RAW_IMAGES_DIR)
    return send_from_directory(image_root, image_path)

@app.route("/api/events/<event_id>/infer", methods=["POST"])
def infer(event_id):
    """Download the image for a cached event and run inference on it."""
    if event_id not in _events_cache:
        return jsonify({"error": "Event not found. Fetch /events first."}), 404
    
    try:
        event = _events_cache[event_id]
        img_data, img_b64 = utils.download_image(event["image_url"])

        heading = event.get("heading")
        length = event.get("estimated_length")

        transformed_img = utils.transform_image(cfg, img_data, heading, device)
        embedding = utils.generate_embedding(cfg, transformed_img, length, model, device)
        top_k_results = utils.similarity_search(cfg, embedding)

        top_k_results_with_image_url = []
        for result in top_k_results["all_results"]:
            raw_path = result["image_path"]
            result["boat_id"] = raw_path.split("_", 1)[0]
            
            try:
                parts = raw_path.split("_", 1)[1].rsplit(".", 1)[0]
                result_event_id = parts 
                
                fetched_event = utils.get_event_by_id(result_event_id)
                
                if fetched_event:
                    result["image_url"] = fetched_event["eventDetails"].get("imageUrl")
                    result["coords"] = [fetched_event["start"]["point"]["lat"], fetched_event["start"]["point"]["lon"]]
                    result["time"] = fetched_event["start"]["time"]
                else:
                    result["image_url"] = None 
                    result["coords"] = [0, 0]
                    result["time"] = "Unknown"
                    app.logger.warning(f"Metadata lookup failed for ID: {result_event_id}")
                    
            except Exception as e:
                app.logger.error(f"Error parsing path {raw_path}: {e}")
                continue

            top_k_results_with_image_url.append(result)

        return jsonify({
            "event": event,
            "query_image": img_b64,
            "all_results": top_k_results_with_image_url,
        }), 200

    except Exception as e:
        app.logger.error(f"Inference failed for event {event_id}: {str(e)}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
