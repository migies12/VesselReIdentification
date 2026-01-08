# Data layout

This template expects a simple CSV-based index with pre-cropped images.

## Folder structure
```
data/
  images/
    123456789_abcd1234.jpg
    123456789_ef567890.jpg
    987654321_11223344.jpg
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
123456789_abcd1234.jpg,123456789,42.7
123456789_ef567890.jpg,123456789,42.7
987654321_11223344.jpg,987654321,18.3
```

## Splits
- Train/val should be ID-disjoint from gallery/query for evaluation.
- Within a split, multiple images per boat are expected.
