"""
paths.py — single source of truth for the output paths of each of our steps
instead of hardcoding things like dataoutput and then reading a hardcoded path, just make everything reference this file
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# repo root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# data: images, + split images from cloud filtering script
# ---------------------------------------------------------------------------

RAW_IMAGES_DIR      = REPO_ROOT / "data" / "images" / "raw"
FILTERED_IMAGES_DIR = REPO_ROOT / "data" / "images" / "filtered"
CLOUDY_EXCLUDED_DIR = REPO_ROOT / "data" / "images" / "cloudyExcluded"
VESSEL_EXCLUDED_DIR = REPO_ROOT / "data" / "images" / "vesselExcluded"

ROTATED_IMAGES_DIR = REPO_ROOT / "data" / "images" / "rotated"
CROPPED_IMAGES_DIR = REPO_ROOT / "data" / "images" / "cropped"
NORMALIZED_IMAGES_DOR = REPO_ROOT / "data" / "images" / "normalized"

# ---------------------------------------------------------------------------
# data, metadata: csvs and other tracking files for data
# ---------------------------------------------------------------------------

METADATA_DIR           = REPO_ROOT / "data" / "metadata"
RAW_METADATA_CSV       = METADATA_DIR / "raw.csv"
FILTERED_METADATA_CSV  = METADATA_DIR / "filtered.csv"
FETCHED_EVENT_IDS_PATH = METADATA_DIR / "eventIds.txt"

# ---------------------------------------------------------------------------
# dataset splits: csvs used by training/gallery/inference
# ---------------------------------------------------------------------------

DATASET_DIR = REPO_ROOT / "data"
TRAIN_CSV   = DATASET_DIR / "train.csv"
VAL_CSV     = DATASET_DIR / "val.csv"
GALLERY_CSV = DATASET_DIR / "gallery.csv"
QUERY_CSV   = DATASET_DIR / "query.csv"

# ---------------------------------------------------------------------------
# model outputs — trained model, FAISS gallery, training stats
# ---------------------------------------------------------------------------

MODEL_DIR           = REPO_ROOT / "model"
MODEL_CHECKPOINT    = "saved_models/best_model_filtered.pt"
FAISS_INDEX_PATH    = MODEL_DIR / "gallery.index"
FAISS_METADATA_PATH = MODEL_DIR / "gallery_metadata.json"
TRAIN_STATS_CSV     = MODEL_DIR / "train_stats.csv"
TRAIN_STATS_JSON    = MODEL_DIR / "train_stats.json"
LOSS_CURVE_PNG      = MODEL_DIR / "loss_curve.png"

# ---------------------------------------------------------------------------
# statistics: visualizations to tell if our model is doing well
# ---------------------------------------------------------------------------

STATISTICS_DIR       = REPO_ROOT / "statistics"
LOSS_CURVES_PNG      = STATISTICS_DIR / "loss_curves.png"
DISTANCE_STATS_PNG   = STATISTICS_DIR / "distance_stats.png"
VALID_FRACTION_PNG   = STATISTICS_DIR / "valid_fraction.png"
TRAINING_SUMMARY_PNG = STATISTICS_DIR / "training_summary.png"
