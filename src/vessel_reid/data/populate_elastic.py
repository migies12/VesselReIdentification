from dotenv import load_dotenv
import os
import pandas as pd
import requests
from tqdm import tqdm

from . import api_helper_elastic, data_utils
from .config import ES_FETCH_DAYS
from vessel_reid.paths import (
    RAW_IMAGES_DIR as IMAGE_DST_PATH,
    RAW_METADATA_CSV as MASTER_CSV_PATH,
    FETCHED_EVENT_IDS_PATH
)


def run(days: int = 30) -> None:
    load_dotenv()
    IMAGE_DST_PATH.mkdir(parents=True, exist_ok=True)

    all_events = api_helper_elastic.get_events_from_elasticsearch(days=days)
    fetched_ids = data_utils.load_fetched_event_ids(FETCHED_EVENT_IDS_PATH)
    new_events = [e for e in all_events if e['eventId'] not in fetched_ids]
    print(f"Skipping {len(all_events) - len(new_events)} already-fetched events")
    if not new_events:
        print("No new events to process")
        return
    
    results = []
    succeeded_event_ids = set()
    failed_count = 0
    skylight_base_url = os.getenv("IMAGE_BASE_URL", "https://cdn.sky-prod-a.skylight.earth")


    for event in tqdm(new_events, desc="Downloading images"):
        mmsi = event["vessels"]["vessel0"]["mmsi"]
        image_url = event["eventDetails"]["imageUrl"]

        if not image_url.startswith("http"):
            image_url = f"{skylight_base_url}/{image_url}"

        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            image_filename = f"{mmsi}_{event['eventId']}.png"
            output_path = IMAGE_DST_PATH / image_filename
            with open(output_path, "wb") as f:
                f.write(image_response.content)

            results.append({
                "image_path": image_filename,
                "boat_id": str(mmsi),
                "length_m": event["eventDetails"].get("estimatedLength", ""),
                "heading": event["eventDetails"].get("heading", "")
            })
            succeeded_event_ids.add(event["eventId"])

        except requests.exceptions.RequestException as e:
            failed_count += 1
            tqdm.write(f"  Failed to download {image_url}: {e}")

    if results:
        df = pd.DataFrame(results)
        header = not MASTER_CSV_PATH.exists()
        df.to_csv(MASTER_CSV_PATH, mode="a", index=False, header=header)
        updated_ids = fetched_ids.union(succeeded_event_ids)
        data_utils.save_event_ids(FETCHED_EVENT_IDS_PATH, updated_ids)

    print(f"\n=== Summary ===")
    print(f"Total images saved: {len(df)}")
    print(f"Failed downloads: {failed_count}")
    print(f"Metadata appended to: {MASTER_CSV_PATH}")

if __name__ == "__main__":
    run(days=ES_FETCH_DAYS)
