"""Fetch vessel images and metadata, saving to CSV and disk.

Reads data.source from configs/shared.yaml to choose between
GraphQL (Skylight API) and Elasticsearch.

Usage:
    python -m vessel_reid.api.populate_data
"""
from pathlib import Path

from dotenv import load_dotenv

from vessel_reid.api import api_helper, elasticsearch_client, graphql_client
from vessel_reid.utils.config import load_config

IMAGE_DST_PATH = Path(__file__).resolve().parent / "../../../data/images"
MASTER_CSV_PATH = IMAGE_DST_PATH.parent / "all_labels.csv"
FETCHED_EVENT_IDS_PATH = IMAGE_DST_PATH.parent / "fetched_event_ids.txt"


if __name__ == "__main__":
    load_dotenv()
    cfg = load_config()

    data_source = cfg.get("data", {}).get("source", "graphql")
    days = cfg.get("data", {}).get("fetch_days", 30)

    if data_source == "elasticsearch":
        events = elasticsearch_client.get_events_from_elasticsearch(days)
    else:
        events = graphql_client.fetch_all_events(days)

    api_helper.download_and_save_events(
        events, IMAGE_DST_PATH, MASTER_CSV_PATH, FETCHED_EVENT_IDS_PATH
    )
