"""Elasticsearch client for fetching vessel events."""
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List
import os

import requests
from tqdm import tqdm


def get_events_from_elasticsearch(days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch AIS-correlated vessel events from Elasticsearch.
    Returns events in the same dict format as the GraphQL client, with filters:
    - event_type: eo_sentinel2
    - correlated: True
    - estimated_length >= 150
    - Only MMSIs with 3+ events
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

    index = os.getenv("ES_INDEX", "event-history")
    verify_ssl = os.getenv("ES_VERIFY_SSL", "true").lower() in ("1", "true", "yes")
    timeout_s = float(os.getenv("ES_TIMEOUT", "60"))

    since = datetime.now(timezone.utc) - timedelta(days=days)

    # First, get all MMSIs with 3+ events using composite aggregation
    print("Fetching MMSIs with 3+ events from Elasticsearch...")
    mmsis_with_multiple_events = set()
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
            f"{es_url.rstrip('/')}/{index}/_search",
            headers=headers,
            auth=auth,
            json=agg_query,
            verify=verify_ssl,
            timeout=timeout_s
        )
        resp.raise_for_status()

        data = resp.json()
        buckets = data.get("aggregations", {}).get("mmsi_page", {}).get("buckets", [])

        for bucket in buckets:
            if bucket.get("doc_count", 0) >= 3:
                mmsis_with_multiple_events.add(bucket["key"]["mmsi"])

        after_key = data.get("aggregations", {}).get("mmsi_page", {}).get("after_key")
        if not after_key:
            break

    print(f"Found {len(mmsis_with_multiple_events)} MMSIs with 3+ events")

    if not mmsis_with_multiple_events:
        return []

    # Now fetch all events for those MMSIs
    all_events = []

    mmsi_list = list(mmsis_with_multiple_events)
    batch_size = 100
    num_batches = math.ceil(len(mmsi_list) / batch_size)

    for i in tqdm(range(0, len(mmsi_list), batch_size), total=num_batches, desc="Fetching events"):
        mmsi_batch = mmsi_list[i:i+batch_size]

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
            f"{es_url.rstrip('/')}/{index}/_search?scroll=2m",
            headers=headers,
            auth=auth,
            json=search_query,
            verify=verify_ssl,
            timeout=timeout_s
        )
        resp.raise_for_status()

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        scroll_id = data.get("_scroll_id")

        all_events.extend(_extract_events(hits))

        while scroll_id and len(hits) > 0:
            scroll_resp = requests.post(
                f"{es_url.rstrip('/')}/_search/scroll",
                headers=headers,
                auth=auth,
                json={"scroll": "2m", "scroll_id": scroll_id},
                verify=verify_ssl,
                timeout=timeout_s
            )
            scroll_resp.raise_for_status()

            data = scroll_resp.json()
            hits = data.get("hits", {}).get("hits", [])
            scroll_id = data.get("_scroll_id")

            all_events.extend(_extract_events(hits))

    print(f"Fetched {len(all_events)} total events from Elasticsearch")
    return all_events


def _extract_events(hits: list) -> list:
    """Transform ES hits into the shared event dict format."""
    events = []
    for hit in hits:
        source = hit["_source"]
        events.append({
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
        })
    return events
