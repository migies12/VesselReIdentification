from datetime import datetime, timedelta, timezone
import math
import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

from .config import MIN_IMAGES_PER_VESSEL

ES_INDEX = "event-history"
ES_VERIFY_SSL = True
ES_TIMEOUT_S = 60.0
ES_BATCH_SIZE = 100  # number of MMSIs per query batch

load_dotenv()


def get_events_from_elasticsearch(days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch AIS-correlated vessel events from Elasticsearch.
    Returns events for MMSIs that have at least MIN_IMAGES_PER_VESSEL events, filtered by:
    - event_type: eo_sentinel2
    - correlated: True
    - estimated_length >= 150
    """
    es_url = os.getenv("ES_URL")
    if not es_url:
        raise ValueError("ES_URL not set in environment")

    username = os.getenv("ES_USERNAME")
    password = os.getenv("ES_PASSWORD")
    api_key = os.getenv("ES_API_KEY")

    headers = {"Content-Type": "application/json"}
    auth = None

    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    elif username and password:
        auth = (username, password)
    else:
        raise ValueError("ES authentication not configured. Set ES_USERNAME/ES_PASSWORD or ES_API_KEY")

    since = datetime.now(timezone.utc) - timedelta(days=days)

    # First, get all MMSIs with MIN_IMAGES_PER_VESSEL+ events using composite aggregation
    print(f"Fetching MMSIs with {MIN_IMAGES_PER_VESSEL}+ events from Elasticsearch...")
    mmsis_with_enough_events = set()
    after_key = None

    while True:
        agg_query = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"event_type": "eo_sentinel2"}},
                        {"term": {"event_details.correlated": True}},
                        {"range": {"event_details.estimated_length": {"gte": 150}}},
                        {"range": {"start.time": {"gte": since.isoformat()}}}
                    ]
                }
            },
            "aggs": {
                "mmsi_page": {
                    "composite": {
                        "size": 1000,
                        "sources": [
                            {"mmsi": {"terms": {"field": "vessels.vessel_0.mmsi"}}}
                        ]
                    }
                }
            }
        }

        if after_key:
            agg_query["aggs"]["mmsi_page"]["composite"]["after"] = after_key

        resp = requests.post(
            f"{es_url.rstrip('/')}/{ES_INDEX}/_search",
            headers=headers,
            auth=auth,
            json=agg_query,
            verify=ES_VERIFY_SSL,
            timeout=ES_TIMEOUT_S,
        )
        resp.raise_for_status()

        data = resp.json()
        buckets = data.get("aggregations", {}).get("mmsi_page", {}).get("buckets", [])

        for bucket in buckets:
            if bucket.get("doc_count", 0) >= MIN_IMAGES_PER_VESSEL:
                mmsis_with_enough_events.add(bucket["key"]["mmsi"])

        after_key = data.get("aggregations", {}).get("mmsi_page", {}).get("after_key")
        if not after_key:
            break

    print(f"Found {len(mmsis_with_enough_events)} MMSIs with {MIN_IMAGES_PER_VESSEL}+ events")

    if not mmsis_with_enough_events:
        return []

    # Now fetch all events for those MMSIs in batches
    all_events = []
    mmsi_list = list(mmsis_with_enough_events)
    num_batches = math.ceil(len(mmsi_list) / ES_BATCH_SIZE)

    for i in tqdm(range(0, len(mmsi_list), ES_BATCH_SIZE), total=num_batches, desc="Fetching events"):
        mmsi_batch = mmsi_list[i:i + ES_BATCH_SIZE]

        search_query = {
            "size": 1000,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"event_type": "eo_sentinel2"}},
                        {"term": {"event_details.correlated": True}},
                        {"range": {"event_details.estimated_length": {"gte": 150}}},
                        {"range": {"start.time": {"gte": since.isoformat()}}},
                        {"terms": {"vessels.vessel_0.mmsi": mmsi_batch}}
                    ]
                }
            },
            "_source": [
                "event_id",
                "vessels.vessel_0.mmsi",
                "event_details.estimated_length",
                "event_details.image_url",
                "event_details.heading"
            ]
        }

        resp = requests.post(
            f"{es_url.rstrip('/')}/{ES_INDEX}/_search?scroll=2m",
            headers=headers,
            auth=auth,
            json=search_query,
            verify=ES_VERIFY_SSL,
            timeout=ES_TIMEOUT_S,
        )
        resp.raise_for_status()

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        scroll_id = data.get("_scroll_id")

        def parse_hit(hit):
            source = hit["_source"]
            return {
                "eventId": source.get("event_id"),
                "vessels": {
                    "vessel0": {
                        "mmsi": source.get("vessels", {}).get("vessel_0", {}).get("mmsi")
                    }
                },
                "eventDetails": {
                    "imageUrl": source.get("event_details", {}).get("image_url"),
                    "estimatedLength": source.get("event_details", {}).get("estimated_length"),
                    "heading": source.get("event_details", {}).get("heading")
                }
            }

        all_events.extend(parse_hit(h) for h in hits)

        while scroll_id and len(hits) > 0:
            scroll_resp = requests.post(
                f"{es_url.rstrip('/')}/_search/scroll",
                headers=headers,
                auth=auth,
                json={"scroll": "2m", "scroll_id": scroll_id},
                verify=ES_VERIFY_SSL,
                timeout=ES_TIMEOUT_S,
            )
            scroll_resp.raise_for_status()

            data = scroll_resp.json()
            hits = data.get("hits", {}).get("hits", [])
            scroll_id = data.get("_scroll_id")
            all_events.extend(parse_hit(h) for h in hits)

    print(f"Fetched {len(all_events)} total events from Elasticsearch")
    return all_events
