import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import config
from vessel_reid.paths import RAW_METADATA_CSV as CSV_PATH


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[df["cloud_coverage"].notna() & (df["cloud_coverage"] != "")]
    df["cloud_coverage"] = df["cloud_coverage"].astype(float)
    return df


def compute_survival(df: pd.DataFrame, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each threshold, compute:
      - proportion of images with cloud_coverage <= threshold
      - proportion of boats that retain >= MIN_IMAGES_PER_VESSEL clean images
    Returns (image_proportions, boat_proportions).
    """
    total_images = len(df)
    total_boats = df["boat_id"].nunique()

    image_props = np.empty(len(thresholds))
    boat_props = np.empty(len(thresholds))

    for i, t in enumerate(thresholds):
        clean = df[df["cloud_coverage"] <= t]
        image_props[i] = len(clean) / total_images

        boat_counts = clean.groupby("boat_id").size()
        qualifying = (boat_counts >= config.MIN_IMAGES_PER_VESSEL).sum()
        boat_props[i] = qualifying / total_boats

    return image_props, boat_props


def main():
    df = load_data()

    if df.empty:
        print("No cloud_coverage data found in CSV. Run filter_clouds.py first.")
        return

    total_images = len(df)
    total_boats = df["boat_id"].nunique()
    print(f"Loaded {total_images} images across {total_boats} boats.")

    thresholds = np.linspace(0, 1, 300)
    image_props, boat_props = compute_survival(df, thresholds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cloud Coverage Analysis", fontsize=14, fontweight="bold")

    # --- Plot 1: Distribution of cloud coverage values ---
    ax1 = axes[0]
    ax1.hist(df["cloud_coverage"], bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax1.axvline(
        config.COVERAGE_THRESHOLD,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Current threshold ({config.COVERAGE_THRESHOLD})",
    )
    ax1.set_xlabel("Cloud Coverage Fraction")
    ax1.set_ylabel("Number of Images")
    ax1.set_title("Cloud Coverage Distribution")
    ax1.legend()

    # --- Plot 2: Survival curves vs threshold ---
    ax2 = axes[1]
    ax2.plot(thresholds, image_props, color="steelblue", linewidth=2, label="Images kept")
    ax2.plot(thresholds, boat_props, color="darkorange", linewidth=2,
             label=f"Boats kept (≥{config.MIN_IMAGES_PER_VESSEL} clean images)")
    ax2.axvline(
        config.COVERAGE_THRESHOLD,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Current threshold ({config.COVERAGE_THRESHOLD})",
    )
    ax2.set_xlabel("Cloud Coverage Threshold")
    ax2.set_ylabel("Proportion Kept")
    ax2.set_title("Images and Boats Kept vs Cloud Coverage Threshold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.legend()

    # Annotate current threshold values on plot 2
    current_img = np.interp(config.COVERAGE_THRESHOLD, thresholds, image_props)
    current_boat = np.interp(config.COVERAGE_THRESHOLD, thresholds, boat_props)
    ax2.annotate(
        f"{current_img:.1%}",
        xy=(config.COVERAGE_THRESHOLD, current_img),
        xytext=(config.COVERAGE_THRESHOLD + 0.03, current_img + 0.04),
        color="steelblue",
        fontsize=9,
    )
    ax2.annotate(
        f"{current_boat:.1%}",
        xy=(config.COVERAGE_THRESHOLD, current_boat),
        xytext=(config.COVERAGE_THRESHOLD + 0.03, current_boat - 0.06),
        color="darkorange",
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
