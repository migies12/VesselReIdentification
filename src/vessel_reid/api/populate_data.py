import api_helper
from dotenv import load_dotenv
from collections import defaultdict
import csv
import os
from pathlib import Path
import requests
from uuid import uuid4

IMAGE_DST_PATH = Path(__file__).resolve().parent / "../../../data/images"
MASTER_CSV_PATH = IMAGE_DST_PATH.parent / "all_labels.csv"
FETCHED_EVENT_IDS_PATH = IMAGE_DST_PATH.parent / "fetched_event_ids.txt"


def load_fetched_event_ids(path: Path) -> set:
    """Load previously fetched event IDs from file"""
    if not path.exists():
        return set()

    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_event_ids(path: Path, event_ids: set) -> None:
    """Append new event IDs to the file"""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing IDs
    existing_ids = load_fetched_event_ids(path)

    # Combine with new IDs
    all_ids = existing_ids.union(event_ids)

    # Write all IDs back to file
    with open(path, "w", encoding="utf-8") as f:
        for event_id in sorted(all_ids):
            f.write(f"{event_id}\n")


def upsert_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                if "image_path" in existing:
                    rows[existing["image_path"]] = existing

    rows[row["image_path"]] = row

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "boat_id", "length_m", "heading"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])

if __name__ == "__main__":
    load_dotenv()

    # Fetch events from Elasticsearch (already filtered to MMSIs with 3+ events)
    all_events = api_helper.get_events_from_elasticsearch(days=30)

    # Track images per vessel
    vessel_images = defaultdict(list)
    for event in all_events:
        mmsi = event['vessels']['vessel0']['mmsi']
        vessel_images[mmsi].append(event)

    # Log statistics
    print(f"\nTotal vessels detected: {len(vessel_images)}")
    print(f"Total events: {len(all_events)}")
    print("\nImages per vessel:")
    for mmsi, events_list in sorted(vessel_images.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  Vessel {mmsi}: {len(events_list)} images")

    # Download images (all vessels already have 3+ images from ES query)
    saved_count = 0

    for mmsi, events_list in vessel_images.items():
        print(f"\nProcessing vessel {mmsi} ({len(events_list)} images)")
        for event in events_list:
            image_response = requests.get(event['eventDetails']['imageUrl'], timeout=30)
            image_response.raise_for_status()

            output_path = IMAGE_DST_PATH / f"{mmsi}_{uuid4().hex}.jpg"
            length_m = event["eventDetails"].get("estimatedLength")
            heading = event["eventDetails"].get("heading")

            if not IMAGE_DST_PATH.exists():
                IMAGE_DST_PATH.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(image_response.content)

            saved_count += 1
            print(f"  Saved {output_path.name}")

            upsert_row(
                MASTER_CSV_PATH,
                {
                    "image_path": output_path.name,
                    "boat_id": str(mmsi),
                    "length_m": "" if length_m is None else length_m,
                    "heading": "" if heading is None else heading,
                },
            )

    print(f"\n=== Summary ===")
    print(f"Total vessels: {len(vessel_images)}")
    print(f"Total images saved: {saved_count}")

    # Save processed event IDs
    event_ids = {event['eventId'] for event in all_events}
    save_event_ids(FETCHED_EVENT_IDS_PATH, event_ids)
    print(f"\nSaved {len(event_ids)} event IDs to {FETCHED_EVENT_IDS_PATH}")
