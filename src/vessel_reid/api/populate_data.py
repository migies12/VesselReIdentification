from vessel_reid.api import api_helper
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
MIN_IMAGES_PER_VESSEL = 3
BACKFILL_LOOKBACK_DAYS = 540
BACKFILL_EVENT_TYPES = ["eo_sentinel2", "eo_landsat_8_9", "sar_sentinel1"]
BACKFILL_MIN_ESTIMATED_LENGTH = 150


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
        fieldnames = ["image_path", "boat_id", "length_m"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])

if __name__ == "__main__":
    load_dotenv()
    access_token = api_helper.get_access_token(os.getenv("SKYLIGHT_USERNAME"), os.getenv("SKYLIGHT_PASSWORD"))
    fetched_event_ids = load_fetched_event_ids(FETCHED_EVENT_IDS_PATH)

    # Fetch all pages of results
    all_events = []
    offset = 0
    limit = 1000

    print("Fetching vessel detections from Skylight API...")
    while True:
        print(f"  Fetching events {offset} to {offset + limit}...")
        response = api_helper.get_recent_correlated_vessels(
            access_token,
            30,
            offset,
            limit=limit,
            event_types=BACKFILL_EVENT_TYPES,
            min_estimated_length=BACKFILL_MIN_ESTIMATED_LENGTH,
        )

        records = response["records"]
        total = response["meta"]["total"]
        all_events.extend(records)

        print(f"  Retrieved {len(records)} events (total available: {total})")

        # Check if we've fetched all results
        if offset + len(records) >= total:
            break

        offset += limit

    print(f"\nFetched {len(all_events)} total events across all pages")

    # Track images per vessel
    vessel_images = defaultdict(list)
    seen_event_ids = set()
    for event in all_events:
        if event["eventId"] in seen_event_ids:
            continue
        seen_event_ids.add(event["eventId"])
        if not event.get("eventDetails") or not event["eventDetails"].get("imageUrl"):
            continue
        mmsi = event['vessels']['vessel0']['mmsi']
        vessel_images[mmsi].append(event)

    # Log statistics
    print(f"\nTotal vessels detected: {len(vessel_images)}")
    print(f"Total events: {len(all_events)}")
    print("\nImages per vessel:")
    for mmsi, events_list in sorted(vessel_images.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  Vessel {mmsi}: {len(events_list)} images")

    # Download images
    saved_count = 0
    filtered_count = 0
    downloaded_event_ids = set()

    def backfill_vessel_events(mmsi: int, target_count: int, existing_event_ids: set) -> list:
        fetched = []
        offset = 0
        while len(fetched) + len(vessel_images[mmsi]) < target_count:
            try:
                response = api_helper.get_recent_correlated_events_for_vessel(
                    access_token,
                    mmsi,
                    BACKFILL_LOOKBACK_DAYS,
                    offset=offset,
                    limit=limit,
                    event_types=BACKFILL_EVENT_TYPES,
                    min_estimated_length=BACKFILL_MIN_ESTIMATED_LENGTH,
                )
            except RuntimeError as exc:
                print(f"  Backfill failed for vessel {mmsi}: {exc}")
                break

            records = response["records"]
            total = response["meta"]["total"]
            if not records:
                break

            for event in records:
                if event["eventId"] in existing_event_ids:
                    continue
                if not event.get("eventDetails") or not event["eventDetails"].get("imageUrl"):
                    continue
                existing_event_ids.add(event["eventId"])
                fetched.append(event)
                if len(fetched) + len(vessel_images[mmsi]) >= target_count:
                    break

            if offset + len(records) >= total:
                break
            offset += limit

        return fetched

    for mmsi, events_list in vessel_images.items():
        if len(events_list) < MIN_IMAGES_PER_VESSEL:
            print(f"\nVessel {mmsi} has only {len(events_list)} images; backfilling...")
            backfilled = backfill_vessel_events(mmsi, MIN_IMAGES_PER_VESSEL, seen_event_ids)
            if backfilled:
                vessel_images[mmsi].extend(backfilled)
                events_list = vessel_images[mmsi]
                print(f"  Added {len(backfilled)} images from backfill (total now {len(events_list)})")

        if len(events_list) < MIN_IMAGES_PER_VESSEL:
            print(f"\nSkipping vessel {mmsi} (only {len(events_list)} images, need {MIN_IMAGES_PER_VESSEL})")
            filtered_count += 1
            continue

        print(f"\nProcessing vessel {mmsi} ({len(events_list)} images)")
        for event in events_list:
            if event["eventId"] in fetched_event_ids:
                continue

            image_response = requests.get(event['eventDetails']['imageUrl'], timeout=30)
            image_response.raise_for_status()

            output_path = IMAGE_DST_PATH / f"{mmsi}_{uuid4().hex}.jpg"
            length_m = event["eventDetails"].get("estimatedLength")

            if not IMAGE_DST_PATH.exists():
                IMAGE_DST_PATH.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(image_response.content)

            saved_count += 1
            downloaded_event_ids.add(event["eventId"])
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

    # Save processed event IDs
    save_event_ids(FETCHED_EVENT_IDS_PATH, downloaded_event_ids)
    print(f"\nSaved {len(downloaded_event_ids)} event IDs to {FETCHED_EVENT_IDS_PATH}")
