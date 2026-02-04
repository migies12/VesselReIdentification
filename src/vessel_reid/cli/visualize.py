"""Visualize training metrics from train_stats.csv."""
import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("--stats-csv", default="outputs/train_stats.csv", help="Path to train_stats.csv")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    return parser.parse_args()


def load_stats(csv_path: str) -> Dict[str, List[float]]:
    """Load training stats from CSV."""
    data: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_arcface_loss": [],
        "train_pos_dist": [],
        "train_neg_dist": [],
        "train_valid_frac": [],
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")

        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            for key in list(data.keys())[1:]:
                val = row.get(key, "")
                if val == "" or val is None:
                    data[key].append(float("nan"))
                else:
                    data[key].append(float(val))

    return data


def plot_loss_curves(data: Dict[str, List[float]], output_path: str, show: bool = False) -> None:
    """Plot training loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data["epoch"], data["train_loss"], "b-o", label="Total Loss", linewidth=2, markersize=4)

    # Only plot arcface if it has non-zero values
    if any(v == v and v != 0.0 for v in data["train_arcface_loss"]):
        ax.plot(data["epoch"], data["train_arcface_loss"], "r-s", label="ArcFace Loss", linewidth=2, markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Over Time", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_distance_stats(data: Dict[str, List[float]], output_path: str, show: bool = False) -> None:
    """Plot positive/negative distance statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Distance plot
    ax1.plot(data["epoch"], data["train_pos_dist"], "g-o", label="Positive Distance", linewidth=2, markersize=4)
    ax1.plot(data["epoch"], data["train_neg_dist"], "r-s", label="Negative Distance", linewidth=2, markersize=4)

    # Add margin visualization
    margin = [n - p for p, n in zip(data["train_pos_dist"], data["train_neg_dist"])]
    ax1.fill_between(data["epoch"], data["train_pos_dist"], data["train_neg_dist"], alpha=0.2, color="blue")

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Distance", fontsize=12)
    ax1.set_title("Triplet Distances Over Time", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Margin plot
    ax2.plot(data["epoch"], margin, "purple", linewidth=2, marker="o", markersize=4)
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax2.fill_between(data["epoch"], margin, 0, where=[m > 0 for m in margin], alpha=0.3, color="green", label="Good margin")
    ax2.fill_between(data["epoch"], margin, 0, where=[m <= 0 for m in margin], alpha=0.3, color="red", label="Bad margin")

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Margin (neg - pos)", fontsize=12)
    ax2.set_title("Embedding Margin Over Time", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_valid_fraction(data: Dict[str, List[float]], output_path: str, show: bool = False) -> None:
    """Plot valid triplet fraction."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data["epoch"], data["train_valid_frac"], "orange", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Ideal (100%)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Valid Fraction", fontsize=12)
    ax.set_title("Valid Triplet Fraction Over Time", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_summary(data: Dict[str, List[float]], output_path: str, show: bool = False) -> None:
    """Create a comprehensive summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(data["epoch"], data["train_loss"], "b-o", label="Total Loss", linewidth=2, markersize=3)
    if any(v == v and v != 0.0 for v in data["train_arcface_loss"]):
        ax.plot(data["epoch"], data["train_arcface_loss"], "r-s", label="ArcFace", linewidth=2, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distances
    ax = axes[0, 1]
    ax.plot(data["epoch"], data["train_pos_dist"], "g-o", label="Positive", linewidth=2, markersize=3)
    ax.plot(data["epoch"], data["train_neg_dist"], "r-s", label="Negative", linewidth=2, markersize=3)
    ax.fill_between(data["epoch"], data["train_pos_dist"], data["train_neg_dist"], alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Distance")
    ax.set_title("Triplet Distances")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Margin
    ax = axes[1, 0]
    margin = [n - p for p, n in zip(data["train_pos_dist"], data["train_neg_dist"])]
    ax.plot(data["epoch"], margin, "purple", linewidth=2, marker="o", markersize=3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.fill_between(data["epoch"], margin, 0, where=[m > 0 for m in margin], alpha=0.3, color="green")
    ax.fill_between(data["epoch"], margin, 0, where=[m <= 0 for m in margin], alpha=0.3, color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Margin (neg - pos)")
    ax.set_title("Embedding Margin")
    ax.grid(True, alpha=0.3)

    # Valid fraction
    ax = axes[1, 1]
    ax.plot(data["epoch"], data["train_valid_frac"], "orange", linewidth=2, marker="o", markersize=3)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valid Fraction")
    ax.set_title("Valid Triplet Fraction")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Summary", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def print_summary(data: Dict[str, List[float]]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Total epochs: {len(data['epoch'])}")
    print(f"Final loss: {data['train_loss'][-1]:.4f}")
    print(f"Best loss: {min(data['train_loss']):.4f} (epoch {data['epoch'][data['train_loss'].index(min(data['train_loss']))]})")
    print(f"Final pos_dist: {data['train_pos_dist'][-1]:.4f}")
    print(f"Final neg_dist: {data['train_neg_dist'][-1]:.4f}")
    print(f"Final margin: {data['train_neg_dist'][-1] - data['train_pos_dist'][-1]:.4f}")
    print(f"Final valid_frac: {data['train_valid_frac'][-1]:.4f}")
    print("=" * 50)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.stats_csv):
        print(f"Error: {args.stats_csv} not found")
        print("Run training first to generate stats, or specify --stats-csv path")
        return

    print(f"Loading stats from {args.stats_csv}")
    data = load_stats(args.stats_csv)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating plots...")
    plot_loss_curves(data, os.path.join(args.output_dir, "loss_curves.png"), args.show)
    plot_distance_stats(data, os.path.join(args.output_dir, "distance_stats.png"), args.show)
    plot_valid_fraction(data, os.path.join(args.output_dir, "valid_fraction.png"), args.show)
    plot_summary(data, os.path.join(args.output_dir, "training_summary.png"), args.show)

    print_summary(data)


if __name__ == "__main__":
    main()
