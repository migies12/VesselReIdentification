"""GraphQL / Skylight API client for fetching vessel events."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import os
from typing import List, Optional

import requests

from vessel_reid.api.api_helper import get_access_token

MIN_IMAGES_PER_VESSEL = 3
BACKFILL_LOOKBACK_DAYS = 540
BACKFILL_EVENT_TYPES = ["eo_sentinel2", "eo_landsat_8_9", "sar_sentinel1"]
BACKFILL_MIN_ESTIMATED_LENGTH = 150
VERBOSE = os.getenv("POPULATE_VERBOSE", "0") == "1"


def get_recent_correlated_vessels(
    access_token: str,
    days: int,
    offset: int = 0,
    limit: int = 1000,
    event_types: Optional[List[str]] = None,
    min_estimated_length: Optional[float] = 150,
):
    """
    Fetch AIS-correlated detections from the Skylight API,
    including the image and associated metadata.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    query = """
        query SearchEventsV2($input: SearchEventsV2Input!) {
            searchEventsV2(input: $input) {
                records {
                    eventId
                    eventType
                    start {
                        time
                        point { lat lon }
                    }
                    end {
                        time
                        point { lat lon }
                    }
                    vessels {
                        vessel0 {
                            mmsi
                            name
                            countryCode
                        }
                    }
                    eventDetails {
                        ... on ImageryMetadataEventDetails {
                            detectionType
                            score
                            estimatedLength
                            frameIds
                            imageUrl
                            orientation
                            heading
                        }
                        ... on ViirsEventDetails {
                            detectionType
                            estimatedLength
                            frameIds
                            imageUrl
                        }
                    }
                }
                meta {
                    total
                }
            }
        }
    """

    if event_types is None:
        event_types = ["eo_sentinel2"]

    event_details = {"detectionType": {"eq": "ais_correlated"}}
    if min_estimated_length is not None:
        event_details["detectionEstimatedLength"] = {"gte": min_estimated_length}

    variables = {
        "input": {
            "eventType": {"inc": event_types},
            "startTime": {"gte": since.isoformat()},
            "eventDetails": event_details,
            "limit": limit,
            "offset": offset,
            "sortBy": "created",
            "sortDirection": "desc"
        }
    }

    response = requests.post(
        os.getenv("GRAPHQL_URL"),
        json={
            "query": query,
            "variables": variables
        },
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if "errors" in data:
        print(f"DEBUG - HTTP Status Code: {response.status_code}")
        print(f"DEBUG - API Error Response: {data}")
        if data.get("data", {}).get("searchEventsV2") is None:
            return {"records": [], "meta": {"total": 0}}
        raise RuntimeError(data["errors"])

    return data["data"]["searchEventsV2"]


def get_recent_correlated_events_for_vessel(
    access_token: str,
    mmsi: int,
    days: int,
    offset: int = 0,
    limit: int = 1000,
    event_types: Optional[List[str]] = None,
    min_estimated_length: Optional[float] = 150,
):
    """Fetch AIS-correlated detections for a specific vessel by MMSI."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    query = """
        query SearchEventsV2($input: SearchEventsV2Input!) {
            searchEventsV2(input: $input) {
                records {
                    eventId
                    eventType
                    start {
                        time
                        point { lat lon }
                    }
                    end {
                        time
                        point { lat lon }
                    }
                    vessels {
                        vessel0 {
                            mmsi
                            name
                            countryCode
                        }
                    }
                    eventDetails {
                        ... on ImageryMetadataEventDetails {
                            detectionType
                            score
                            estimatedLength
                            frameIds
                            imageUrl
                            orientation
                            heading
                        }
                        ... on ViirsEventDetails {
                            detectionType
                            estimatedLength
                            frameIds
                            imageUrl
                        }
                    }
                }
                meta {
                    total
                }
            }
        }
    """

    if event_types is None:
        event_types = ["eo_sentinel2"]

    event_details = {"detectionType": {"eq": "ais_correlated"}}
    if min_estimated_length is not None:
        event_details["detectionEstimatedLength"] = {"gte": min_estimated_length}

    variables = {
        "input": {
            "eventType": {"inc": event_types},
            "startTime": {"gte": since.isoformat()},
            "eventDetails": event_details,
            "vesselMain": {"mmsi": {"eq": str(mmsi)}},
            "limit": limit,
            "offset": offset,
            "sortBy": "created",
            "sortDirection": "desc",
        }
    }

    response = requests.post(
        os.getenv("GRAPHQL_URL"),
        json={
            "query": query,
            "variables": variables
        },
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if "errors" in data:
        print(f"DEBUG - HTTP Status Code: {response.status_code}")
        print(f"DEBUG - API Error Response: {data}")
        if data.get("data", {}).get("searchEventsV2") is None:
            return {"records": [], "meta": {"total": 0}}
        raise RuntimeError(data["errors"])

    return data["data"]["searchEventsV2"]


def fetch_all_events(days: int) -> list:
    """Fetch all events via GraphQL, backfilling vessels with <3 images.

    Returns a flat list of events ready for downloading.
    """
    access_token = get_access_token(
        os.getenv("SKYLIGHT_USERNAME"), os.getenv("SKYLIGHT_PASSWORD")
    )

    # Paginate through all events
    all_events = []
    offset = 0
    limit = 1000

    print("Fetching vessel detections from Skylight API...")
    while True:
        print(f"  Fetching events {offset} to {offset + limit}...")
        response = get_recent_correlated_vessels(
            access_token,
            days,
            offset,
            limit=limit,
            event_types=BACKFILL_EVENT_TYPES,
            min_estimated_length=BACKFILL_MIN_ESTIMATED_LENGTH,
        )

        records = response["records"]
        total = response["meta"]["total"]
        all_events.extend(records)

        print(f"  Retrieved {len(records)} events (total available: {total})")

        if offset + len(records) >= total:
            break

        offset += limit

    print(f"\nFetched {len(all_events)} total events across all pages")

    # Group by vessel, dedup, filter out events without images
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

    print(f"\nTotal vessels detected: {len(vessel_images)}")
    print(f"Total events: {len(all_events)}")
    if VERBOSE:
        print("\nImages per vessel:")
        for mmsi, events_list in sorted(vessel_images.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  Vessel {mmsi}: {len(events_list)} images")

    # Backfill vessels with fewer than MIN_IMAGES_PER_VESSEL
    def backfill_vessel_events(mmsi, target_count, existing_event_ids):
        fetched = []
        bf_offset = 0
        while len(fetched) + len(vessel_images[mmsi]) < target_count:
            try:
                resp = get_recent_correlated_events_for_vessel(
                    access_token,
                    mmsi,
                    BACKFILL_LOOKBACK_DAYS,
                    offset=bf_offset,
                    limit=limit,
                    event_types=BACKFILL_EVENT_TYPES,
                    min_estimated_length=BACKFILL_MIN_ESTIMATED_LENGTH,
                )
            except RuntimeError as exc:
                print(f"  Backfill failed for vessel {mmsi}: {exc}")
                break

            recs = resp["records"]
            bf_total = resp["meta"]["total"]
            if not recs:
                break

            for event in recs:
                if event["eventId"] in existing_event_ids:
                    continue
                if not event.get("eventDetails") or not event["eventDetails"].get("imageUrl"):
                    continue
                existing_event_ids.add(event["eventId"])
                fetched.append(event)
                if len(fetched) + len(vessel_images[mmsi]) >= target_count:
                    break

            if bf_offset + len(recs) >= bf_total:
                break
            bf_offset += limit

        return fetched

    filtered_count = 0
    result_events = []
    total_vessels = len(vessel_images)

    for idx, (mmsi, events_list) in enumerate(vessel_images.items(), start=1):
        if idx == 1 or idx % 50 == 0 or idx == total_vessels:
            print(f"\nProcessing vessels {idx}/{total_vessels}...")
        if len(events_list) < MIN_IMAGES_PER_VESSEL:
            if VERBOSE:
                print(f"\nVessel {mmsi} has only {len(events_list)} images; backfilling...")
            backfilled = backfill_vessel_events(mmsi, MIN_IMAGES_PER_VESSEL, seen_event_ids)
            if backfilled:
                vessel_images[mmsi].extend(backfilled)
                events_list = vessel_images[mmsi]
                if VERBOSE:
                    print(f"  Added {len(backfilled)} images from backfill (total now {len(events_list)})")

        if len(events_list) < MIN_IMAGES_PER_VESSEL:
            if VERBOSE:
                print(f"\nSkipping vessel {mmsi} (only {len(events_list)} images, need {MIN_IMAGES_PER_VESSEL})")
            filtered_count += 1
            continue

        result_events.extend(events_list)

    print(f"Vessels filtered (< {MIN_IMAGES_PER_VESSEL} images): {filtered_count}")
    print(f"Vessels kept: {total_vessels - filtered_count}")
    print(f"Total events to download: {len(result_events)}")

    return result_events
