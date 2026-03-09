import csv
from pathlib import Path
import shutil

CSV_FIELDNAMES = ["image_path", "boat_id", "length_m", "heading", "cloud_coverage"]


def load_fetched_event_ids(path: Path) -> set:
    """Load previously fetched event IDs from file"""
    if not path.exists():
        return set()

    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_event_ids(path: Path, event_ids: set) -> None:
    """Append new event IDs to the file"""
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_fetched_event_ids(path)
    all_ids = existing_ids.union(event_ids)

    with open(path, "w", encoding="utf-8") as f:
        for event_id in sorted(all_ids):
            f.write(f"{event_id}\n")


def load_csv(csv_path: Path) -> dict:
    """Load CSV into a dict keyed by image_path."""
    if not csv_path.exists():
        return {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["image_path"]: row for row in reader if "image_path" in row}


def write_csv(csv_path: Path, rows: dict) -> None:
    """Write a full rows dict (keyed by image_path) to the CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])


def batch_upsert_rows(csv_path: Path, rows_to_write: dict) -> None:
    """Merge a batch of rows into the CSV to make downloading images less fucking slow"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_csv(csv_path)
    for image_path, row in rows_to_write.items():
        existing = rows.get(image_path, {})
        rows[image_path] = {**existing, **row}
    write_csv(csv_path, rows)


def upsert_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                if "image_path" in existing:
                    rows[existing["image_path"]] = existing

    # Merge with existing row so fields not in the new row (e.g. cloud_coverage) are preserved
    existing = rows.get(row["image_path"], {})
    rows[row["image_path"]] = {**existing, **row}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, restval="", extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])

def setup_dryrun_folder(path):
    """
    Helper pre-processing scripts
    Prepares the output folder specified by `path` by deleting existing data inside the folder
    Used by `filter_clouds` and `crop_water`
    """
    if path.exists():
        print(f"Clearing existing files in {path}")
        shutil.rmtree(path, ignore_errors=True)

    path.mkdir(parents=True, exist_ok=True)
