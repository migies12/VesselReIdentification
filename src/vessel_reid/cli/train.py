import argparse
import os
from typing import Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from vessel_reid.data.dataset import DataConfig, TripletDataset
from vessel_reid.models.reid_model import ReIDModel
from vessel_reid.utils.config import load_config
from vessel_reid.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vessel ReID model")
    parser.add_argument("--config", required=True, help="Path to train config YAML")
    return parser.parse_args()


def move_batch(batch: Tuple, device: torch.device):
    (a_img, a_len), (p_img, p_len), (n_img, n_len) = batch
    a_img = a_img.to(device)
    p_img = p_img.to(device)
    n_img = n_img.to(device)
    if a_len is not None:
        a_len = a_len.to(device)
        p_len = p_len.to(device)
        n_len = n_len.to(device)
    return (a_img, a_len), (p_img, p_len), (n_img, n_len)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int,
    scaler: GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        (a_img, a_len), (p_img, p_len), (n_img, n_len) = move_batch(batch, device)
        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            a_emb = model(a_img, a_len)
            p_emb = model(p_img, p_len)
            n_emb = model(n_img, n_len)
            loss = criterion(a_emb, p_emb, n_emb)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if (step + 1) % log_every == 0:
            avg = total_loss / (step + 1)
            tqdm.write(f"step {step + 1}: loss {avg:.4f}")
    return total_loss / max(len(loader), 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(
        csv_path=cfg["data"]["train_csv"],
        image_root=cfg["data"]["image_root"],
        image_size=cfg["data"]["image_size"],
        use_length=cfg["data"]["use_length"],
        length_mean=cfg["data"]["length_mean"],
        length_std=cfg["data"]["length_std"],
        rotate_by_direction=cfg["data"].get("rotate_by_direction", False),
    )
    dataset = TripletDataset(data_cfg)
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.TripletMarginLoss(margin=cfg["train"]["margin"], p=2)
    scaler = GradScaler()

    for epoch in range(cfg["train"]["epochs"]):
        loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
            cfg["train"]["log_every"],
            scaler,
        )
        print(f"epoch {epoch + 1}/{cfg['train']['epochs']}: loss {loss:.4f}")

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)
    checkpoint_path = os.path.join(cfg["train"]["output_dir"], cfg["train"]["checkpoint_name"])
    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved model to {checkpoint_path}")


if __name__ == "__main__":
    main()
