from datetime import datetime, timedelta, timezone
import os
import requests

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

def get_recent_correlated_vessels(access_token: str, days: int):
    """
    Fetch the 30 most recent AIS-correlated detections from the Skylight API,
    including the image and associated metadata

    access_token: A valid Skylight API access token obtained via `get_access_token()`
    days: Number of days to look back from the current time (UTC). Only detections with
            timestamps greater than or equal to now - days will be returned
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
            "limit": 100,
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