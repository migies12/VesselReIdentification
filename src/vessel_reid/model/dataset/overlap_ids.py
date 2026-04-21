import argparse
import os
from typing import Set

import pandas as pd


def load_ids(csv_path: str, id_col: str) -> Set[str]:
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"column '{id_col}' not found in {csv_path}")
    return set(df[id_col].astype(str).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Find overlapping boat IDs between two CSVs.")
    parser.add_argument("--query-csv", required=True, help="Path to query CSV")
    parser.add_argument("--gallery-csv", required=True, help="Path to gallery CSV")
    parser.add_argument("--id-col", default="boat_id", help="ID column name (default: boat_id)")
    parser.add_argument("--out-dir", default=None, help="Optional output dir for filtered CSVs")
    args = parser.parse_args()

    query_ids = load_ids(args.query_csv, args.id_col)
    gallery_ids = load_ids(args.gallery_csv, args.id_col)
    overlap_ids = query_ids.intersection(gallery_ids)

    print(f"query IDs: {len(query_ids)}")
    print(f"gallery IDs: {len(gallery_ids)}")
    print(f"overlap IDs: {len(overlap_ids)}")

    if not overlap_ids:
        print("No overlapping IDs found.")
        return

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        query_df = pd.read_csv(args.query_csv)
        gallery_df = pd.read_csv(args.gallery_csv)
        query_df[args.id_col] = query_df[args.id_col].astype(str)
        gallery_df[args.id_col] = gallery_df[args.id_col].astype(str)

        query_out = query_df[query_df[args.id_col].isin(overlap_ids)]
        gallery_out = gallery_df[gallery_df[args.id_col].isin(overlap_ids)]

        query_out_path = os.path.join(args.out_dir, "query_overlap.csv")
        gallery_out_path = os.path.join(args.out_dir, "gallery_overlap.csv")
        query_out.to_csv(query_out_path, index=False)
        gallery_out.to_csv(gallery_out_path, index=False)

        print(f"wrote {query_out_path} with {len(query_out)} rows")
        print(f"wrote {gallery_out_path} with {len(gallery_out)} rows")


if __name__ == "__main__":
    main()
