import api_helper_elastic
from collections import defaultdict
from dotenv import load_dotenv
import os
import requests
from tqdm import tqdm

import data_utils
from vessel_reid.paths import RAW_IMAGES_DIR, RAW_METADATA_CSV, FETCHED_EVENT_IDS_PATH

IMAGE_DST_PATH  = RAW_IMAGES_DIR
MASTER_CSV_PATH = RAW_METADATA_CSV


def run(days: int = 30) -> None:
    load_dotenv()

    all_events = api_helper_elastic.get_events_from_elasticsearch(days=days)

    fetched_ids = data_utils.load_fetched_event_ids(FETCHED_EVENT_IDS_PATH)
    new_events = [e for e in all_events if e['eventId'] not in fetched_ids]
    print(f"Skipping {len(all_events) - len(new_events)} already-fetched events")

    vessel_images = defaultdict(list)
    for event in new_events:
        mmsi = event['vessels']['vessel0']['mmsi']
        vessel_images[mmsi].append(event)

    saved_count = 0
    failed_count = 0
    succeeded_event_ids = set()
    pending_rows = {}
    all_downloads = [(mmsi, event) for mmsi, events_list in vessel_images.items() for event in events_list]

    skylight_base_url = os.getenv("IMAGE_BASE_URL", "https://cdn.sky-prod-a.skylight.earth")

    IMAGE_DST_PATH.mkdir(parents=True, exist_ok=True)

    for mmsi, event in tqdm(all_downloads, desc="Downloading images"):
        image_url = event['eventDetails']['imageUrl']
        if not image_url.startswith("http"):
            image_url = f"{skylight_base_url}/{image_url}"
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            failed_count += 1
            tqdm.write(f"  Failed to download {image_url}: {e}")
            continue

        output_path = IMAGE_DST_PATH / f"{mmsi}_{event['eventId']}.png"
        length_m = event["eventDetails"].get("estimatedLength")
        heading = event["eventDetails"].get("heading")

        with open(output_path, "wb") as f:
            f.write(image_response.content)

        saved_count += 1
        succeeded_event_ids.add(event['eventId'])
        pending_rows[output_path.name] = {
            "image_path": output_path.name,
            "boat_id": str(mmsi),
            "length_m": "" if length_m is None else length_m,
            "heading": "" if heading is None else heading,
        }

        if saved_count % 1000 == 0:
            # batch the csv writes bc downloading is getting painfully slow from csv writes and its making me mad 
            # maybe we should in fact get a db 
            data_utils.batch_upsert_rows(MASTER_CSV_PATH, pending_rows)
            pending_rows = {}
            data_utils.save_event_ids(FETCHED_EVENT_IDS_PATH, succeeded_event_ids)

    if pending_rows:
        data_utils.batch_upsert_rows(MASTER_CSV_PATH, pending_rows)

    print(f"\n=== Summary ===")
    print(f"Total vessels: {len(vessel_images)}")
    print(f"Total images saved: {saved_count}")
    print(f"Failed downloads: {failed_count}")

    data_utils.save_event_ids(FETCHED_EVENT_IDS_PATH, succeeded_event_ids)
    print(f"Saved {len(succeeded_event_ids)} event IDs to {FETCHED_EVENT_IDS_PATH}")
