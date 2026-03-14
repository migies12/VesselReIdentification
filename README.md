# Vessel Re-Identification

Training a vessel ReID model on pre-cropped satellite images and deploying a FAISS gallery for similarity search.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data layout
See `src/vessel_reid/model/README.md` for the full module structure, config options, and CLI reference.

## Typical workflow
1) Prepare ID-disjoint splits (train/val/gallery/query).
2) Train the embedding model with triplet loss.
3) Build the FAISS gallery from known boats.
4) Run inference with new images and a similarity threshold.

Generate splits:
```bash
python -m vessel_reid.model.dataset.split_ids --csv data/metadata/raw.csv --out-dir data/
```
Or generate splits directly from image filenames:
```bash
python -m vessel_reid.model.dataset.split_ids --image-dir data/images/filtered --out-dir data/
```

## Commands
Train:
```bash
python -m vessel_reid.model.training.training --config configs/train.yaml
```

Build gallery:
```bash
python -m vessel_reid.model.inference.build_gallery --config configs/gallery.yaml
```

Query:
```bash
python -m vessel_reid.model.inference.infer --config configs/inference.yaml --image path/to/query.jpg --length-m 42.7
```

Visualize training metrics:
```bash
python -m vessel_reid.model.training.training_curves --stats-csv model/train_stats.csv --output-dir statistics/
```
## Notes
- Assumes each image contains one boat (no detection stage).
- For evaluation, keep boat IDs disjoint across train/val/test.
- In deployment, it is expected to keep multiple images per boat in the gallery.
