import os
import sys
import json
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

INDEX = os.getenv("ES_INDEX", "event-history")

def build_query(after: Optional[Dict[str, Any]] = None, use_keyword_event_type: bool = False) -> Dict[str, Any]:
    event_field = "event_type.keyword" if use_keyword_event_type else "event_type"

    q: Dict[str, Any] = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"term": {event_field: "eo_sentinel2"}},
                    {"term": {"event_details.correlated": True}},
                    {"range": {"event_details.estimated_length": {"gte": 150}}},
                ]
            }
        },
        "aggs": {
            "mmsi_page": {
                "composite": {
                    "size": 1000,
                    "sources": [
                        {"mmsi": {"terms": {"field": "vessels.vessel_0.mmsi"}}}
                    ],
                }
            }
        },
    }

    if after is not None:
        q["aggs"]["mmsi_page"]["composite"]["after"] = after

    return q

def main() -> int:
    # IMPORTANT: This must be the Elasticsearch endpoint, NOT the Kibana URL.
    # Example Elastic Cloud ES URL often looks like:
    # https://<deployment>.es.us-west1.gcp.cloud.es.io:9243
    es_url = os.getenv("ES_URL")
    if not es_url:
        print("Missing ES_URL. Set it to your Elasticsearch endpoint (not Kibana).")
        print('Example: export ES_URL="https://<deployment>.es.us-west1.gcp.cloud.es.io:9243"')
        return 2

    # Auth: either API key or basic auth
    api_key = os.getenv("ES_API_KEY")  # value like: "base64encoded=="
    username = os.getenv("ES_USERNAME")
    password = os.getenv("ES_PASSWORD")

    headers = {"Content-Type": "application/json"}
    auth = None

    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    elif username and password:
        auth = (username, password)
    else:
        print("Missing auth. Set ES_API_KEY or ES_USERNAME + ES_PASSWORD.")
        return 2

    verify_ssl = os.getenv("ES_VERIFY_SSL", "true").lower() in ("1", "true", "yes")
    timeout_s = float(os.getenv("ES_TIMEOUT", "30"))

    search_url = f"{es_url.rstrip('/')}/{INDEX}/_search"

    # Some clusters map event_type as text; try normal field first, then fallback to .keyword
    use_keyword_event_type = False

    total_mmsi_over_3 = 0
    pages = 0
    after_key = None

    while True:
        body = build_query(after=after_key, use_keyword_event_type=use_keyword_event_type)

        resp = requests.post(search_url, headers=headers, auth=auth, json=body, verify=verify_ssl, timeout=timeout_s)

        # If event_type is text and term query fails silently, you might get 0 hits across all pages.
        # If ES errors on mapping, we'll retry using event_type.keyword once.
        if resp.status_code >= 400:
            # Try keyword fallback once if we haven't already
            if not use_keyword_event_type:
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": resp.text}
                print("Initial query failed; retrying with event_type.keyword. Error was:")
                print(json.dumps(err, indent=2) if isinstance(err, dict) else err)
                use_keyword_event_type = True
                continue

            # Otherwise, fail
            try:
                print(json.dumps(resp.json(), indent=2))
            except Exception:
                print(resp.text)
            resp.raise_for_status()

        data = resp.json()
        agg = data.get("aggregations", {}).get("mmsi_page", {})
        buckets = agg.get("buckets", [])

        # Count MMSIs whose bucket doc_count >= 3
        total_mmsi_over_3 += sum(1 for b in buckets if b.get("doc_count", 0) >= 3)

        pages += 1
        after_key = agg.get("after_key")

        # Stop when no after_key (no more pages)
        if not after_key:
            break

        # Optional progress
        if pages % 10 == 0:
            print(f"Processed {pages} pages... current count={total_mmsi_over_3}", file=sys.stderr)

    print(total_mmsi_over_3)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
