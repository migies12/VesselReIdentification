import argparse
import os
import random

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ID-disjoint splits")
    parser.add_argument("--csv", required=True, help="Master CSV with all labeled images")
    parser.add_argument("--out-dir", required=True, help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val split ratio")
    parser.add_argument("--gallery", type=float, default=0.1, help="Gallery split ratio")
    parser.add_argument("--query", type=float, default=0.1, help="Query split ratio")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(args.train + args.val + args.gallery + args.query - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    df = pd.read_csv(args.csv)
    boat_ids = sorted(df["boat_id"].astype(str).unique().tolist())

    random.seed(args.seed)
    random.shuffle(boat_ids)

    n = len(boat_ids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_gallery = int(n * args.gallery)

    train_ids = set(boat_ids[:n_train])
    val_ids = set(boat_ids[n_train : n_train + n_val])
    gallery_ids = set(boat_ids[n_train + n_val : n_train + n_val + n_gallery])
    query_ids = set(boat_ids[n_train + n_val + n_gallery :])

    splits = {
        "train": train_ids,
        "val": val_ids,
        "gallery": gallery_ids,
        "query": query_ids,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in splits.items():
        split_df = df[df["boat_id"].astype(str).isin(ids)]
        split_df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        print(f"wrote {name}.csv with {len(split_df)} rows")


if __name__ == "__main__":
    main()
