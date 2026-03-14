# Model

This directory contains everything related to the ReID model — training, inference, data loading, and preprocessing.

## Structure

```
model/
├── dataset/            # Data loading and train/val/gallery/query splitting
├── image_processing/   # Crop, rotate, and background normalization
├── models/             # ResNet50 backbone + embedding head
├── training/           # Training loop and loss curves visualization
├── inference/          # FAISS gallery building and query inference
└── utils/              # Config loading, seeding, metrics
```

### dataset/
- `dataset.py` — dataset classes, transforms, and PK batch sampler
- `split_ids.py` — splits a labeled CSV into ID-disjoint train/val/gallery/query CSVs
- `overlap_ids.py` — checks for vessel ID overlap between query and gallery splits

### image_processing/
- `crop.py` — crops the outer border of an image
- `rotate.py` — rotates images to normalize vessel heading
- `normalize.py` — replaces the water background with gray using K-Means clustering

### models/
- `reid_model.py` — ResNet50 backbone that outputs a 512-dim L2-normalized embedding, with optional vessel length fusion

### training/
- `training.py` — main training script (triplet loss + ArcFace, PK sampling, mixed precision)
- `training_curves.py` — generates loss/distance plots from the training stats CSV

### inference/
- `faiss_index.py` — utilities for building, saving, loading, and searching a FAISS index
- `build_gallery.py` — encodes all gallery images and writes a FAISS index to disk
- `infer.py` — queries a single image against the gallery and returns matches above a similarity threshold

### utils/
- `config.py` — YAML config loader
- `seed.py` — deterministic seeding
- `metrics.py` — evaluation metrics

---

## Workflow

```
1. split_ids.py      → produce train/val/gallery/query CSVs
2. training.py       → train the model
3. build_gallery.py  → encode gallery images into FAISS index
4. infer.py          → query an image against the gallery
```

---

## CLI Reference

### `split_ids.py`

```bash
python -m vessel_reid.model.dataset.split_ids \
  --out-dir data/ \
  [--csv data/metadata/raw.csv] \
  [--image-dir data/images/filtered] \
  [--train 0.7] [--val 0.1] [--gallery 0.1] [--query 0.1] \
  [--seed 1337]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--out-dir` | Yes | — | Where to write the output CSVs |
| `--csv` | No | — | Master CSV with all labeled images |
| `--image-dir` | No | — | If provided, filters the CSV to only images present in this directory |
| `--train` | No | 0.7 | Train split ratio |
| `--val` | No | 0.1 | Val split ratio |
| `--gallery` | No | 0.1 | Gallery split ratio |
| `--query` | No | 0.1 | Query split ratio |
| `--seed` | No | 1337 | Random seed |

---

### `overlap_ids.py`

```bash
python -m vessel_reid.model.dataset.overlap_ids \
  --query-csv data/query.csv \
  --gallery-csv data/gallery.csv \
  [--id-col boat_id] \
  [--out-dir data/]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--query-csv` | Yes | — | Path to query CSV |
| `--gallery-csv` | Yes | — | Path to gallery CSV |
| `--id-col` | No | `boat_id` | Column name for vessel IDs |
| `--out-dir` | No | — | If provided, writes overlap-removed CSVs here |

---

### `training.py`

```bash
python -m vessel_reid.model.training.training --config configs/train.yaml
```

Outputs to `model/`: `reid_model.pt`, `train_stats.csv`, `train_stats.json`, `loss_curve.png`.

**Key config options (`configs/train.yaml`):**

| Option | Description |
|--------|-------------|
| `data.use_length` | Fuse vessel length into the embedding |
| `data.compute_length_stats` | Compute length mean/std from train CSV automatically |
| `data.rotate/crop/normalize` | Enable preprocessing steps |
| `data.augment` | Enable brightness/contrast/rotation augmentations during training |
| `data.pk_sampler.p` / `.k` | Classes per batch / samples per class |
| `train.loss` | `"triplet"`, `"arcface"`, or `"combined"` |
| `train.triplet.distance` | `"cosine"` or `"euclidean"` |
| `train.triplet.margin` | Triplet loss margin |
| `train.arcface_scale` / `arcface_margin` | ArcFace hyperparameters |
| `train.triplet_weight` / `arcface_weight` | Relative loss weights when using `combined` |

---

### `build_gallery.py`

```bash
python -m vessel_reid.model.inference.build_gallery --config configs/gallery.yaml
```

Outputs to `model/`: `gallery.index`, `gallery_metadata.json`.

---

### `infer.py`

```bash
python -m vessel_reid.model.inference.infer \
  --config configs/inference.yaml \
  --image path/to/query.jpg \
  [--length-m 210.5] \
  [--heading 45.0]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to inference config YAML |
| `--image` | Yes | Path to query image |
| `--length-m` | No | Vessel length in meters |
| `--heading` | No | Vessel heading in degrees (0–360), used for rotation preprocessing |

Key config options in `configs/inference.yaml`: `query.similarity_threshold` (default 0.7) and `query.top_k` (default 5).
