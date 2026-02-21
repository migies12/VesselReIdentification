import base64
from flask import Flask, jsonify, request
import os
import torch

from ..utils.config import load_config
from . import utils

app = Flask(__name__)

# Global Initialization
# Load the config
api_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(api_dir, "..", "..", ".."))
cfg = load_config(os.path.join(root, "configs", "inference.yaml"))

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = utils.load_model(cfg, device, os.path.join(os.path.dirname(__file__), "model.pt"))

@app.route("/")
def home():
    return "<h2>Vessel Reidentification</p>"

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

