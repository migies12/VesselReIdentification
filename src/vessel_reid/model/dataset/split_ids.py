import argparse
import os
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ID-disjoint splits")
    parser.add_argument("--csv", help="Master CSV with all labeled images")
    parser.add_argument("--image-dir", help="Directory of images named <boat_id>_<hash>.*")
    parser.add_argument("--out-dir", required=True, help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val split ratio")
    parser.add_argument("--gallery", type=float, default=0.1, help="Gallery split ratio (by ID)")
    parser.add_argument("--query", type=float, default=0.1, help="Query split ratio (by ID)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    return parser.parse_args()


def build_df_from_images(image_dir: Path) -> pd.DataFrame:
    image_dir = image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")

    allowed_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    rows = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_exts:
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        boat_id = stem.split("_", 1)[0]
        rows.append({"image_path": path.name, "boat_id": boat_id})

    if not rows:
        raise ValueError(f"no images found in {image_dir} with <boat_id>_<hash>.* naming")

    return pd.DataFrame(rows)


def split_gallery_query_within_ids(df: pd.DataFrame, ids: set, query_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    gallery_rows = []
    query_rows = []

    for boat_id, group in df[df["boat_id"].astype(str).isin(ids)].groupby("boat_id"):
        rows = group.sample(frac=1.0, random_state=rng.randint(0, 1_000_000))
        count = len(rows)
        if count == 1:
            # Only one image: keep it in gallery so it can be retrieved.
            gallery_rows.append(rows)
            continue

        n_query = max(1, int(round(count * query_ratio)))
        n_query = min(n_query, count - 1)
        query_rows.append(rows.iloc[:n_query])
        gallery_rows.append(rows.iloc[n_query:])

    gallery_df = pd.concat(gallery_rows, ignore_index=True) if gallery_rows else pd.DataFrame(columns=df.columns)
    query_df = pd.concat(query_rows, ignore_index=True) if query_rows else pd.DataFrame(columns=df.columns)
    return gallery_df, query_df


def main() -> None:
    args = parse_args()
    if abs(args.train + args.val + args.gallery + args.query - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    if not args.csv and not args.image_dir:
        raise ValueError("provide either --csv or --image-dir")

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = build_df_from_images(Path(args.image_dir))

    boat_ids = sorted(df["boat_id"].astype(str).unique().tolist())

    random.seed(args.seed)
    random.shuffle(boat_ids)

    n = len(boat_ids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_gallery = int(n * args.gallery)

    train_ids = set(boat_ids[:n_train])
    val_ids = set(boat_ids[n_train : n_train + n_val])
    eval_ids = set(boat_ids[n_train + n_val :])

    # Build train/val as ID-disjoint, but split gallery/query within the same IDs.
    train_df = df[df["boat_id"].astype(str).isin(train_ids)]
    val_df = df[df["boat_id"].astype(str).isin(val_ids)]

    gallery_ratio = args.gallery
    query_ratio = args.query
    if gallery_ratio + query_ratio == 0:
        raise ValueError("gallery + query ratio must be > 0")
    query_ratio = query_ratio / (gallery_ratio + query_ratio)

    gallery_df, query_df = split_gallery_query_within_ids(df, eval_ids, query_ratio, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    print(f"wrote train.csv with {len(train_df)} rows")
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    print(f"wrote val.csv with {len(val_df)} rows")
    gallery_df.to_csv(os.path.join(args.out_dir, "gallery.csv"), index=False)
    print(f"wrote gallery.csv with {len(gallery_df)} rows")
    query_df.to_csv(os.path.join(args.out_dir, "query.csv"), index=False)
    print(f"wrote query.csv with {len(query_df)} rows")


if __name__ == "__main__":
    main()
