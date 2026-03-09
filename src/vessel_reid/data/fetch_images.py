import argparse
import importlib

from .config import DEFAULT_DATA_SOURCE

SOURCES = {
    "gql": "vessel_reid.data.populate_skylight",
    "es": "vessel_reid.data.populate_elastic",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch vessel images from a data source")
    parser.add_argument(
        "--source",
        choices=SOURCES.keys(),
        default=DEFAULT_DATA_SOURCE,
        help=f"Data source to fetch from: 'gql' (Skylight GraphQL API) or 'es' (Elasticsearch). Default: {DEFAULT_DATA_SOURCE}",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to look back (default: 30)",
    )
    args = parser.parse_args()

    populate = importlib.import_module(SOURCES[args.source])
    populate.run(days=args.days)
