import argparse
import csv
import json
import os
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from vessel_reid.data.dataset import DataConfig, LabeledImageDataset, PKBatchSampler
from vessel_reid.models.reid_model import ReIDModel
from vessel_reid.utils.config import load_config
from vessel_reid.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vessel ReID model")
    parser.add_argument("--config", default="configs/shared.yaml", help="Path to config YAML")
    return parser.parse_args()


def move_batch(batch: Tuple, device: torch.device):
    images, lengths, labels = batch
    images = images.to(device)
    lengths = lengths.to(device) if lengths is not None else None
    if torch.is_tensor(labels):
        labels = labels.to(device)
    else:
        labels = torch.tensor(labels, device=device)
    labels = labels.long()
    return images, lengths, labels


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3, distance: str = "cosine", softplus: bool = True) -> None:
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.softplus = softplus

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        n = embeddings.size(0)
        if n <= 1:
            return embeddings.sum() * 0.0, {"pos_dist": 0.0, "neg_dist": 0.0, "valid_frac": 0.0}

        distance = self.distance.lower()
        if distance == "cosine":
            sim = torch.matmul(embeddings, embeddings.t())
            dist = 1.0 - sim
        elif distance in ("euclidean", "euclidian"):
            dot = torch.matmul(embeddings, embeddings.t())
            sq = torch.diag(dot)
            dist = sq.unsqueeze(1) - 2 * dot + sq.unsqueeze(0)
            dist = torch.clamp(dist, min=0.0)
            dist = torch.sqrt(dist + 1e-12)
        else:
            raise ValueError(f"unsupported distance: {self.distance}")

        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())
        mask_pos.fill_diagonal_(False)
        mask_neg = labels.ne(labels.t())

        pos_dist = dist.clone()
        pos_dist[~mask_pos] = -1e9
        hardest_pos, _ = pos_dist.max(dim=1)

        neg_dist = dist.clone()
        neg_dist[~mask_neg] = 1e9
        hardest_neg, _ = neg_dist.min(dim=1)

        valid = (mask_pos.sum(dim=1) > 0) & (mask_neg.sum(dim=1) > 0)
        if valid.any():
            raw = hardest_pos - hardest_neg + self.margin
            losses = torch.nn.functional.softplus(raw) if self.softplus else torch.relu(raw)
            loss = losses[valid].mean()
            pos_mean = hardest_pos[valid].mean().item()
            neg_mean = hardest_neg[valid].mean().item()
            valid_frac = valid.float().mean().item()
        else:
            loss = embeddings.sum() * 0.0
            pos_mean = 0.0
            neg_mean = 0.0
            valid_frac = 0.0

        return loss, {"pos_dist": pos_mean, "neg_dist": neg_mean, "valid_frac": valid_frac}


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.5) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings are expected to be L2-normalized
        cosine = F.linear(embeddings, F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_logit = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = cosine * (1.0 - one_hot) + target_logit * one_hot
        return output * self.scale


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BatchHardTripletLoss,
    arcface_head: Optional[ArcFaceHead],
    arcface_weight: float,
    triplet_weight: float,
    use_arcface: bool,
    use_triplet: bool,
    device: torch.device,
    use_amp: bool,
    log_every: int,
    scaler: GradScaler,
) -> Tuple[float, float, float, float, float]:
    model.train()
    if use_arcface and arcface_head is None:
        raise ValueError("arcface_head must be provided when use_arcface=true")
    total_loss = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_valid = 0.0
    total_arcface = 0.0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images, lengths, labels = move_batch(batch, device)
        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            embeddings = model(images, lengths)
            triplet_loss = torch.tensor(0.0, device=device)
            stats = {"pos_dist": 0.0, "neg_dist": 0.0, "valid_frac": 0.0}
            if use_triplet:
                triplet_loss, stats = criterion(embeddings, labels)

            arcface_loss = torch.tensor(0.0, device=device)
            if use_arcface:
                logits = arcface_head(embeddings, labels)
                arcface_loss = F.cross_entropy(logits, labels)

            if use_triplet and use_arcface:
                loss = triplet_weight * triplet_loss + arcface_weight * arcface_loss
            elif use_arcface:
                loss = arcface_loss
            else:
                loss = triplet_weight * triplet_loss

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_pos += stats["pos_dist"]
        total_neg += stats["neg_dist"]
        total_valid += stats["valid_frac"]
        total_arcface += arcface_loss.item() if use_arcface else 0.0
        if (step + 1) % log_every == 0:
            avg = total_loss / (step + 1)
            pos = total_pos / (step + 1)
            neg = total_neg / (step + 1)
            valid = total_valid / (step + 1)
            if use_arcface:
                arc = total_arcface / (step + 1)
                tqdm.write(
                    f"step {step + 1}: loss {avg:.4f} arc {arc:.4f} pos {pos:.3f} neg {neg:.3f} valid {valid:.2f}"
                )
            else:
                tqdm.write(f"step {step + 1}: loss {avg:.4f} pos {pos:.3f} neg {neg:.3f} valid {valid:.2f}")
    denom = max(len(loader), 1)
    arcface_avg = total_arcface / denom if use_arcface else 0.0
    return total_loss / denom, total_pos / denom, total_neg / denom, total_valid / denom, arcface_avg


def append_stats(path: str, row: Dict[str, float]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_stats_json(path: str, stats: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def maybe_plot_metrics(csv_path: str, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = []
    metrics: Dict[str, list] = {
        "train_loss": [],
        "train_arcface_loss": [],
        "train_pos_dist": [],
        "train_neg_dist": [],
        "train_valid_frac": [],
    }
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        for row in reader:
            epochs.append(int(row["epoch"]))
            for key in metrics:
                val = row.get(key)
                if val is None or val == "":
                    metrics[key].append(float("nan"))
                else:
                    metrics[key].append(float(val))

    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    axes[0].plot(epochs, metrics["train_loss"], marker="o", label="train_loss")
    if any(v == v and v != 0.0 for v in metrics["train_arcface_loss"]):
        axes[0].plot(epochs, metrics["train_arcface_loss"], marker="o", label="arcface_loss")
    axes[0].set_title("loss")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, metrics["train_pos_dist"], marker="o", label="pos_dist")
    axes[1].plot(epochs, metrics["train_neg_dist"], marker="o", label="neg_dist")
    axes[1].set_title("distance stats")
    axes[1].set_ylabel("distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, metrics["train_valid_frac"], marker="o", label="valid_frac")
    axes[2].set_title("valid fraction")
    axes[2].set_ylabel("fraction")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].axis("off")

    fig.suptitle("training metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def compute_length_stats(csv_path: str) -> Tuple[float, float, int]:
    count = 0
    mean = 0.0
    m2 = 0.0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "length_m" not in reader.fieldnames:
            raise ValueError("length_m column not found in training CSV")
        for row in reader:
            val = row.get("length_m")
            if val is None or val == "":
                continue
            try:
                x = float(val)
            except ValueError:
                continue
            count += 1
            delta = x - mean
            mean += delta / count
            m2 += delta * (x - mean)

    if count == 0:
        raise ValueError("no valid length_m values found in training CSV")

    var_pop = m2 / count
    std = var_pop**0.5
    return mean, std, count


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    length_mean = cfg["data"].get("length_mean")
    length_std = cfg["data"].get("length_std")
    if cfg["data"]["use_length"] and cfg["data"].get("compute_length_stats", False):
        length_mean, length_std, count = compute_length_stats(cfg["data"]["train_csv"])
        print(f"computed length stats from train.csv: mean={length_mean:.4f} std={length_std:.4f} count={count}")

    data_cfg = DataConfig(
        csv_path=cfg["data"]["train_csv"],
        image_root=cfg["data"]["image_root"],
        image_size=cfg["data"]["image_size"],
        use_length=cfg["data"]["use_length"],
        length_mean=float(length_mean),
        length_std=float(length_std),
        rotate_by_direction=cfg.get("rotate_by_direction", False),
        crop_ratio=cfg.get("crop_ratio", 0.707),
        augment=cfg["data"].get("augment", True),
    )
    dataset = LabeledImageDataset(data_cfg)

    pk_cfg = cfg["data"].get("pk_sampler", {})
    use_pk = pk_cfg.get("enabled", True)
    if use_pk:
        p = pk_cfg.get("p", 16)
        k = pk_cfg.get("k", 4)
        batch_sampler = PKBatchSampler(
            dataset.indices_by_id,
            p=p,
            k=k,
            batches_per_epoch=pk_cfg.get("batches_per_epoch"),
            seed=cfg["seed"],
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            persistent_workers=True if cfg["data"]["num_workers"] > 0 else False,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            persistent_workers=True if cfg["data"]["num_workers"] > 0 else False,
            drop_last=True,
        )

    model = ReIDModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=cfg["model"]["embedding_dim"],
        use_length=cfg["model"]["use_length"],
        length_embed_dim=cfg["model"]["length_embed_dim"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    loss_mode = cfg["train"].get("loss", "triplet")
    use_triplet = loss_mode in ("triplet", "combined")
    use_arcface = loss_mode in ("arcface", "combined")
    arcface_weight = float(cfg["train"].get("arcface_weight", 1.0))
    triplet_weight = float(cfg["train"].get("triplet_weight", 1.0))

    triplet_cfg = cfg["train"].get("triplet", {})
    criterion = BatchHardTripletLoss(
        margin=float(triplet_cfg.get("margin", cfg["train"]["margin"])),
        distance=str(triplet_cfg.get("distance", "cosine")),
        softplus=bool(triplet_cfg.get("softplus", True)),
    )
    arcface_head = None
    if use_arcface:
        arcface_head = ArcFaceHead(
            embedding_dim=cfg["model"]["embedding_dim"],
            num_classes=len(dataset.id_to_index),
            scale=float(cfg["train"].get("arcface_scale", 30.0)),
            margin=float(cfg["train"].get("arcface_margin", 0.5)),
        ).to(device)

    params = list(model.parameters())
    if use_arcface:
        params += list(arcface_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)
    stats_csv = os.path.join(cfg["train"]["output_dir"], "train_stats.csv")
    stats_json = os.path.join(cfg["train"]["output_dir"], "train_stats.json")
    stats_plot = os.path.join(cfg["train"]["output_dir"], "loss_curve.png")
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        if use_pk:
            loader.batch_sampler.set_epoch(epoch)
        loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            arcface_head,
            arcface_weight,
            triplet_weight,
            use_arcface,
            use_triplet,
            device,
            use_amp,
            cfg["train"]["log_every"],
            scaler,
        )
        train_loss, pos_dist, neg_dist, valid_frac, arcface_loss = loss
        print(
            f"epoch {epoch + 1}/{cfg['train']['epochs']}: "
            f"loss {train_loss:.4f} arc {arcface_loss:.4f} pos {pos_dist:.3f} neg {neg_dist:.3f} valid {valid_frac:.2f}"
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_arcface_loss": arcface_loss,
            "train_pos_dist": pos_dist,
            "train_neg_dist": neg_dist,
            "train_valid_frac": valid_frac,
        }
        history.append(row)
        append_stats(stats_csv, row)
        save_stats_json(stats_json, history)
        maybe_plot_metrics(stats_csv, stats_plot)

    checkpoint_path = os.path.join(cfg["train"]["output_dir"], cfg["train"]["checkpoint_name"])
    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved model to {checkpoint_path}")


if __name__ == "__main__":
    main()
