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
        fieldnames = ["image_path", "boat_id", "length_m"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])

if __name__ == "__main__":
    load_dotenv()
    access_token = api_helper.get_access_token(os.getenv("SKYLIGHT_USERNAME"), os.getenv("SKYLIGHT_PASSWORD"))
    events = api_helper.get_recent_correlated_vessels(access_token, 30)

    # Track images per vessel
    vessel_images = defaultdict(list)
    for event in events["records"]:
        mmsi = event['vessels']['vessel0']['mmsi']
        vessel_images[mmsi].append(event)

    # Log statistics
    print(f"\nTotal vessels detected: {len(vessel_images)}")
    print(f"Total events: {len(events['records'])}")
    print("\nImages per vessel:")
    for mmsi, events_list in sorted(vessel_images.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  Vessel {mmsi}: {len(events_list)} images")

    # Download images
    MIN_IMAGES_PER_VESSEL = 3
    saved_count = 0
    filtered_count = 0

    for mmsi, events_list in vessel_images.items():
        if len(events_list) < MIN_IMAGES_PER_VESSEL:
            print(f"\nSkipping vessel {mmsi} (only {len(events_list)} images, need {MIN_IMAGES_PER_VESSEL})")
            filtered_count += 1
            continue

        print(f"\nProcessing vessel {mmsi} ({len(events_list)} images)")
        for event in events_list:
            image_response = requests.get(event['eventDetails']['imageUrl'], timeout=30)
            image_response.raise_for_status()

            output_path = IMAGE_DST_PATH / f"{mmsi}_{uuid4().hex}.jpg"
            length_m = event["eventDetails"].get("estimatedLength")

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
                },
            )

    print(f"\n=== Summary ===")
    print(f"Total vessels: {len(vessel_images)}")
    print(f"Vessels filtered (< {MIN_IMAGES_PER_VESSEL} images): {filtered_count}")
    print(f"Vessels saved: {len(vessel_images) - filtered_count}")
    print(f"Total images saved: {saved_count}")
