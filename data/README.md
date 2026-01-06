# Data layout

This template expects a simple CSV-based index with pre-cropped images.

## Folder structure
```
data/
  images/
    boat_0001_day1.jpg
    boat_0001_day5.jpg
    boat_0002_day2.jpg
  train.csv
  val.csv
  gallery.csv
  query.csv
```

## CSV schema
Each CSV should contain the following columns:
- `image_path` (relative to `image_root`)
- `boat_id` (string or integer ID)
- `length_m` (float, optional if `use_length=false`)

Example:
```csv
image_path,boat_id,length_m
boat_0001_day1.jpg,1,42.7
boat_0001_day5.jpg,1,42.7
boat_0002_day2.jpg,2,18.3
```

## Splits
- Train/val should be ID-disjoint from gallery/query for evaluation.
- Within a split, multiple images per boat are expected.
