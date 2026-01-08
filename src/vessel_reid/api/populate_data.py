import api_helper
from dotenv import load_dotenv
import os
from pathlib import Path
import requests
from uuid import uuid4

IMAGE_DST_PATH = Path(__file__).resolve().parent / "../../../data/"

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

        with open(output_path, "wb") as f:
            f.write(image_response.content)
            
        print(f"Saved image to {output_path}")
