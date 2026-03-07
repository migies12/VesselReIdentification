from datetime import datetime, timedelta, timezone
import os
import requests
from typing import List, Optional

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

def get_recent_correlated_vessels(access_token: str, days: int, min_length: int = 150):
    """
    Fetch AIS-correlated detections from the Skylight API,
    including the image and associated metadata

    Fetches data for each day separately to avoid the 10,000 event API limit,
    ensuring we can accumulate sufficient data for triplet loss training.

    access_token: A valid Skylight API access token obtained via `get_access_token()`
    days: Number of days to look back from the current time (UTC).
    
    Returns:
        A dict with 'records' and 'meta' keys, containing accumulated results from all days
    """
    all_records = []
    total_count = 0
    
    for delta in range(days):
        end_time = datetime.now(timezone.utc) - timedelta(days=delta)
        start_time = end_time - timedelta(days=1)

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
                "startTime": {"gte": start_time.isoformat(), "lt": end_time.isoformat()},
                "eventDetails": {
                    "detectionType": {"eq": "ais_correlated"},
                    "detectionEstimatedLength": {"gte": min_length}
                },
                "limit": 2000,
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
        if "errors" in data or data.get("data", {}).get("searchEventsV2") is None:
            print(f"Warning: Skylight API error for day {delta}, skipping. Response: {data.get('errors')}")
            continue

        events = data["data"]["searchEventsV2"]
        records = events.get("records", [])
        total = events.get("meta", {}).get("total", 0)

        all_records.extend(records)
        total_count += total

        print(f"Day {delta}: Retrieved {len(records)} events (total for this day: {total})")

    return {
        "records": all_records,
        "meta": {
            "total": total_count
        }
    }

def get_recent_correlated_events_for_vessel(
    access_token: str,
    mmsi: int,
    days: int,
    offset: int = 0,
    limit: int = 1000,
    event_types: Optional[List[str]] = None,
    min_estimated_length: Optional[float] = 150,
):
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

def get_event(access_token: str, event_id: str):
    """
    Fetch a specific event by ID
    """
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
            "eventId": {"eq": event_id}
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
    response = response.json()
    records = response["data"]["searchEventsV2"]["records"]
    return records[0] if records else None
