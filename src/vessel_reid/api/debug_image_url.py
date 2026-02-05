"""Fetch a single Sentinel-2 correlated event from the Skylight GraphQL API and print it."""
import json
import os
from dotenv import load_dotenv
from vessel_reid.api import api_helper

load_dotenv()

access_token = api_helper.get_access_token(
    os.getenv("SKYLIGHT_USERNAME"), os.getenv("SKYLIGHT_PASSWORD")
)
response = api_helper.get_recent_correlated_vessels(access_token, days=30, offset=0)

record = response["records"][0]
print(json.dumps(record, indent=2))
