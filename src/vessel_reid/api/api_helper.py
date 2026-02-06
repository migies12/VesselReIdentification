"""Shared helpers for vessel event fetching and image downloading."""
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm


def get_access_token(username: str, password: str) -> str:
    """
    Requests and returns a valid access_token for the Skylight API for the given Skylight credentials.
    Access tokens are valid for 24 hours.
    """
    query = f'{{getToken(username: "{username}", password: "{password}") {{access_token expires_in}}}}'

    response = requests.post(
        os.getenv("GRAPHQL_URL"),
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    response.raise_for_status()

    data = response.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])

    token_info = data["data"]["getToken"]
    return token_info["access_token"]


def load_fetched_event_ids(path: Path) -> set:
    """Load previously fetched event IDs from file."""
    if not path.exists():
        return set()

    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_event_ids(path: Path, event_ids: set) -> None:
    """Merge new event IDs with existing ones and write to file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_fetched_event_ids(path)
    all_ids = existing_ids.union(event_ids)

    with open(path, "w", encoding="utf-8") as f:
        for event_id in sorted(all_ids):
            f.write(f"{event_id}\n")


def upsert_row(csv_path: Path, row: dict) -> None:
    """Insert or update a row in the CSV, keyed by image_path."""
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
        fieldnames = ["image_path", "boat_id", "length_m", "heading", "event_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])


def download_and_save_events(
    events: list,
    image_dst: Path,
    csv_path: Path,
    event_ids_path: Path,
) -> None:
    """Download images for a list of events and save metadata to CSV.

    Skips events that have already been fetched. Handles both full URLs
    and relative paths (prefixed with IMAGE_BASE_URL env var).
    """
    fetched_ids = load_fetched_event_ids(event_ids_path)
    new_events = [e for e in events if e["eventId"] not in fetched_ids]
    print(f"Skipping {len(events) - len(new_events)} already-fetched events")

    # Group by vessel for summary
    vessel_images = defaultdict(list)
    for event in new_events:
        mmsi = event["vessels"]["vessel0"]["mmsi"]
        vessel_images[mmsi].append(event)

    saved_count = 0
    failed_count = 0
    succeeded_event_ids = set()

    skylight_base_url = os.getenv(
        "IMAGE_BASE_URL", "https://cdn.sky-prod-a.skylight.earth"
    )
    image_dst.mkdir(parents=True, exist_ok=True)

    all_downloads = [
        (mmsi, event)
        for mmsi, event_list in vessel_images.items()
        for event in event_list
    ]

    for mmsi, event in tqdm(all_downloads, desc="Downloading images"):
        image_url = event["eventDetails"]["imageUrl"]
        if not image_url.startswith("http"):
            image_url = f"{skylight_base_url}/{image_url}"

        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            failed_count += 1
            tqdm.write(f"  Failed to download {image_url}: {e}")
            continue

        safe_event_id = re.sub(r"[^\w\-]", "_", event["eventId"])
        output_path = image_dst / f"{mmsi}_{safe_event_id}.png"
        length_m = event["eventDetails"].get("estimatedLength")
        heading = event["eventDetails"].get("heading")

        with open(output_path, "wb") as f:
            f.write(image_response.content)

        saved_count += 1
        succeeded_event_ids.add(event["eventId"])

        upsert_row(
            csv_path,
            {
                "image_path": output_path.name,
                "boat_id": str(mmsi),
                "length_m": "" if length_m is None else length_m,
                "heading": "" if heading is None else heading,
                "event_id": event["eventId"],
            },
        )

        if saved_count % 1000 == 0:
            save_event_ids(event_ids_path, succeeded_event_ids)

    print(f"\n=== Summary ===")
    print(f"Total vessels: {len(vessel_images)}")
    print(f"Total images saved: {saved_count}")
    print(f"Failed downloads: {failed_count}")

    save_event_ids(event_ids_path, succeeded_event_ids)
    print(f"Saved {len(succeeded_event_ids)} event IDs to {event_ids_path}")
