import api_helper
from dotenv import load_dotenv
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
    events = api_helper.get_recent_correlated_vessels(access_token, 1)

    for event in events["records"]:
        mmsi = event['vessels']['vessel0']['mmsi']
        print(f"Detected event for vessel {mmsi} at {event['start']['time']}")
        image_response = requests.get(event['eventDetails']['imageUrl'], timeout=30)
        image_response.raise_for_status()

        output_path = IMAGE_DST_PATH / f"{mmsi}_{uuid4().hex}.jpg"
        length_m = event["eventDetails"].get("estimatedLength")

        if not IMAGE_DST_PATH.exists():
            IMAGE_DST_PATH.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(image_response.content)
            
        print(f"Saved image to {output_path}")

        upsert_row(
            MASTER_CSV_PATH,
            {
                "image_path": output_path.name,
                "boat_id": str(mmsi),
                "length_m": "" if length_m is None else length_m,
            },
        )
