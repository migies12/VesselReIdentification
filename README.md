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
Or generate splits directly from image filenames:
```bash
python -m vessel_reid.cli.split_ids --image-dir data/images --out-dir data
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

---

## Changelog - Branch: `mica`

### `src/vessel_reid/cli/train.py`
- **Change**: Added mixed precision training with `torch.cuda.amp`
- **Why**: Gets 2-3x speedup on A100 GPU by using half-precision where possible

- **Change**: Added `pin_memory=True` and `persistent_workers=True` to DataLoader
- **Why**: Makes data loading faster by keeping workers alive and pinning memory

### `configs/train.yaml`
- **Change**: Increased `batch_size` from 32 to 128
- **Why**: A100 has 40GB of VRAM, was barely using any of it with batch size 32

- **Change**: Increased `num_workers` from 4 to 8
- **Why**: Load data faster with more parallel workers

### `src/vessel_reid/utils/seed.py`
- **Change**: Set `cudnn.benchmark=True` and `cudnn.deterministic=False`
- **Why**: Lets cuDNN find the fastest algorithms for this hardware

### `src/vessel_reid/api/api_helper.py`
- **Change**: Added `get_events_from_elasticsearch()` function
- **Why**: Query Elasticsearch directly instead of GraphQL API for better performance and filtering

- **Change**: Elasticsearch query filters to MMSIs with 3+ events upfront
- **Why**: More efficient than fetching all events and filtering afterward

### `src/vessel_reid/api/populate_data.py`
- **Change**: Switched from GraphQL API to Elasticsearch for data fetching
- **Why**: Direct database access is faster and allows better filtering by MMSI count

- **Change**: Removed client-side MIN_IMAGES_PER_VESSEL filtering
- **Why**: Elasticsearch query already filters to vessels with 3+ events

- **Change**: Save fetched event IDs to `data/fetched_event_ids.txt`
- **Why**: Track which events were fetched for future reproducibility
