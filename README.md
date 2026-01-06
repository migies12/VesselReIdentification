# Vessel Re-Identification Template

Template repository for training a vessel ReID model on pre-cropped satellite images and deploying a FAISS gallery for similarity search.

## What this template provides
- PyTorch ReID model with a ResNet backbone and an embedding head
- Optional metadata fusion (e.g., vessel length) via a small MLP
- Triplet-loss training loop (anchor/positive/negative)
- Scripts to build a FAISS gallery and run inference
- Clear data layout expectations

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data layout
See `data/README.md` for the expected CSV format and folder structure.

## Typical workflow
1) Prepare ID-disjoint splits (train/val/gallery/query).
2) Train the embedding model with triplet loss.
3) Build the FAISS gallery from known boats.
4) Run inference with new images and a similarity threshold.

Generate splits:
```bash
python -m vessel_reid.cli.split_ids --csv data/all_labels.csv --out-dir data
```
## Commands
Train:
```bash
python -m vessel_reid.cli.train --config configs/train.yaml
```

Build gallery:
```bash
python -m vessel_reid.cli.build_gallery --config configs/gallery.yaml
```

Query:
```bash
python -m vessel_reid.cli.infer --config configs/inference.yaml --image path/to/query.jpg --length-m 42.7
```

## Notes
- This template assumes each image contains one boat (no detection stage).
- For evaluation, keep boat IDs disjoint across train/val/test.
- In deployment, it is expected to keep multiple images per boat in the gallery.
