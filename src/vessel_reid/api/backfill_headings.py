"""
Backfill the heading column in all_labels.csv by fetching orientation data
from the Skylight GraphQL API or Elasticsearch using stored event IDs.

Usage:
    python -m vessel_reid.api.backfill_headings
"""
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from vessel_reid.api import api_helper, elasticsearch_client, graphql_client
from vessel_reid.utils.config import load_config

CSV_PATH = Path(__file__).resolve().parent / "../../../data/all_labels.csv"


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["image_path", "boat_id", "length_m", "heading", "event_id"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_headings_graphql(event_ids: set, mmsis: set, days: int) -> dict:
    """Fetch orientation for events via GraphQL.

    Queries per-MMSI and returns a dict mapping event_id -> orientation.
    """
    access_token = api_helper.get_access_token(
        os.getenv("SKYLIGHT_USERNAME"), os.getenv("SKYLIGHT_PASSWORD")
    )

    lookup = {}
    for mmsi in tqdm(sorted(mmsis), desc="Fetching headings (GraphQL)"):
        offset = 0
        limit = 1000
        while True:
            try:
                response = graphql_client.get_recent_correlated_events_for_vessel(
                    access_token,
                    int(mmsi),
                    days,
                    offset=offset,
                    limit=limit,
                )
            except RuntimeError as exc:
                tqdm.write(f"  Failed for MMSI {mmsi}: {exc}")
                break

            records = response.get("records", [])
            total = response.get("meta", {}).get("total", 0)
            if not records:
                break

            for event in records:
                eid = event.get("eventId")
                if eid not in event_ids:
                    continue
                details = event.get("eventDetails")
                if not details:
                    continue
                heading = details.get("heading")
                if heading is not None:
                    lookup[eid] = heading

            if offset + len(records) >= total:
                break
            offset += limit

    return lookup


def fetch_headings_elasticsearch(event_ids: set, mmsis: set, days: int) -> dict:
    """Fetch heading for events via Elasticsearch.

    Returns a dict mapping event_id -> heading.
    """
    all_events = elasticsearch_client.get_events_from_elasticsearch(days=days)

    lookup = {}
    for event in all_events:
        eid = event.get("eventId")
        if eid not in event_ids:
            continue
        heading = event.get("eventDetails", {}).get("heading")
        if heading is not None:
            lookup[eid] = heading

    return lookup


def main() -> None:
    load_dotenv()
    cfg = load_config()

    data_source = cfg.get("data", {}).get("source", "graphql")
    days = cfg.get("data", {}).get("fetch_days", 30)

    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    rows = read_csv(CSV_PATH)
    print(f"Read {len(rows)} rows from {CSV_PATH}")

    # Find rows that have an event_id but no heading
    needs_heading = [r for r in rows if r.get("event_id") and not r.get("heading")]
    if not needs_heading:
        without_event_id = sum(1 for r in rows if not r.get("event_id") and not r.get("heading"))
        if without_event_id:
            print(f"{without_event_id} rows are missing heading but have no event_id — cannot backfill these.")
        else:
            print("All rows already have heading data.")
        return

    event_ids = {r["event_id"] for r in needs_heading}
    mmsis = {r["boat_id"] for r in needs_heading}
    print(f"{len(needs_heading)} rows need heading ({len(mmsis)} vessels, {len(event_ids)} events)")

    if data_source == "elasticsearch":
        lookup = fetch_headings_elasticsearch(event_ids, mmsis, days)
    else:
        lookup = fetch_headings_graphql(event_ids, mmsis, days)

    print(f"Got heading data for {len(lookup)} events")

    updated = 0
    for row in rows:
        if row.get("heading"):
            continue
        eid = row.get("event_id")
        if not eid:
            continue
        heading = lookup.get(eid)
        if heading is not None:
            row["heading"] = heading
            updated += 1

    write_csv(CSV_PATH, rows)
    print(f"Updated {updated}/{len(needs_heading)} rows with heading data")


if __name__ == "__main__":
    main()
