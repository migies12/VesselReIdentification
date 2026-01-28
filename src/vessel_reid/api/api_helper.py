from datetime import datetime, timedelta, timezone
import math
import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def get_access_token(username: str, password: str) -> str:
    """
    Requests and returns a valid access_token for the Skylight API for the given Skylight credentials
    Access tokens are valid for 24 hours
    Usage of an access token:
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    """
    # Note: Skylight API requires credentials to be embedded directly in the query,
    # not passed as variables

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

def get_recent_correlated_vessels(access_token: str, days: int, offset: int = 0):
    """
    Fetch AIS-correlated detections from the Skylight API,
    including the image and associated metadata

    access_token: A valid Skylight API access token obtained via `get_access_token()`
    days: Number of days to look back from the current time (UTC). Only detections with
            timestamps greater than or equal to now - days will be returned
    offset: Pagination offset for fetching results beyond the first page
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
                            vesselType
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
                        }
                    }
                }
                meta {
                    total
                }
            }
        }
    """

    variables = {
        "input": {
            "eventType": {"inc": ["eo_sentinel2"]},
            "startTime": {"gte": since.isoformat()},
            "eventDetails": {
                "detectionType": {"eq": "ais_correlated"},
                "detectionEstimatedLength": {"gte": 150}
            },
            "limit": 1000,
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
        raise RuntimeError(data["errors"])

    return data["data"]["searchEventsV2"]


def get_events_from_elasticsearch(days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch AIS-correlated vessel events from Elasticsearch.
    Returns events grouped by MMSI with filters:
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

    # Query in batches of MMSIs to avoid URL length issues
    mmsi_list = list(mmsis_with_multiple_events)
    batch_size = 100
    num_batches = math.ceil(len(mmsi_list) / batch_size)

    for i in tqdm(range(0, len(mmsi_list), batch_size), total=num_batches, desc="Fetching events"):
        mmsi_batch = mmsi_list[i:i+batch_size]

        # Use scroll API for large result sets
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

        # Process first batch
        for hit in hits:
            source = hit["_source"]
            # Transform ES format to match GraphQL format expected by populate_data.py
            all_events.append({
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

        # Continue scrolling if needed
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

            for hit in hits:
                source = hit["_source"]
                all_events.append({
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


    print(f"Fetched {len(all_events)} total events from Elasticsearch")
    return all_events