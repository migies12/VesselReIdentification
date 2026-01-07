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
    query = """
        query getToken($username: String!, $password: String!) {
            getToken(username: $username, password: $password) {
                access_token
                expires_in
            }
        }
    """
    variables = {
        "username": username,
        "password": password
    }

    response = requests.post(
        os.getenv("GRAPHQL_URL"),
        json={"query": query, "variables": variables},
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
    Fetch AIS-correlated detections from the Skylight API within a recent time window,
    including the image and associated metadata
    
    access_token: A valid Skylight API access token obtained via `get_access_token()`
    days: Number of days to look back from the current time (UTC). Only detections with
            timestamps greater than or equal to now - days will be returned
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    query = """
        query Events($since: DateTime!) {
            events(
                filter: {
                    event_type: { eq: "standard_rendezvous" }
                    start: { gte: $since }
                }
            ) {
                event_id
                event_type
                start
                end
                vessels
                event_details
                user_params
            }
        }
    """

    variables = {
        "since": since.isoformat()
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
        raise RuntimeError(data["errors"])
    
    return data["data"]["events"]