import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vessel_reid.model.dataset.dataset import DataConfig, SingleImageDataset
from vessel_reid.model.inference.faiss_index import load_index, load_metadata
from vessel_reid.model.models.reid_model import ReIDModel
from vessel_reid.model.utils.metrics import (
    compute_rankk_accuracy,
    compute_map,
    compute_f1_at_threshold,
    find_optimal_threshold,
    compute_confidence_margin,
    compute_separation,
    compute_aggregate_score,
)


def encode_dataset(
    model: ReIDModel,
    dataset_cfg: DataConfig,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """
    runs images through the model and gets embeddings
    """
    dataset = SingleImageDataset(dataset_cfg)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_embeddings = []
    all_ids = []

    model.eval()
    with torch.no_grad():
        for images, lengths, boat_ids, _image_paths, _length_values in tqdm(loader, desc="encoding"):
            images = images.to(device)
            if lengths is not None:
                lengths = lengths.to(device)
            embeddings = model(images, lengths)
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend([str(b) for b in boat_ids])

    return np.concatenate(all_embeddings, axis=0), all_ids


def build_similarity_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """
    similarity matrix of query pics vs gallery pics
    shape: num_queries x num_gallery. Higher = more similar.
    """
    return query_embeddings @ gallery_embeddings.T


def evaluate(
    cfg: dict,
    run_dir: Path,
    wandb_run=None,
) -> dict:
    """
    loads model + gallery from run_dir, encodes query set, computes all metrics.
    saves results to run_dir/metrics.json and returns the metrics dict.

    cfg must have "model" and "query" sections.
    pass None to skip wandb logging. Just handy for us to view stats. 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = ReIDModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=cfg["model"]["embedding_dim"],
        use_length=cfg["model"]["use_length"],
        length_embed_dim=cfg["model"]["length_embed_dim"],
        pretrained=False,
    ).to(device)
    state = torch.load(str(run_dir / "reid_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # load gallery embeddings + IDs from faiss index and metadata
    index = load_index(str(run_dir / "gallery.index"))
    gallery_meta = load_metadata(str(run_dir / "gallery_metadata.json"))
    gallery_embeddings = index.reconstruct_n(0, index.ntotal)
    gallery_ids = np.array([m["boat_id"] for m in gallery_meta])

    # encode query set
    q = cfg["query"]
    query_data_cfg = DataConfig(
        csv_path=q["csv_path"],
        image_root=q["image_root"],
        image_size=q["image_size"],
        use_length=q["use_length"],
        length_mean=q["length_mean"],
        length_std=q["length_std"],
        rotate=q.get("rotate", False),
        crop=q.get("crop", False),
        normalize=q.get("normalize", False),
    )
    query_embeddings, query_ids_list = encode_dataset(model, query_data_cfg, device)
    query_ids = np.array(query_ids_list)

    sim_matrix = build_similarity_matrix(query_embeddings, gallery_embeddings)

    threshold = cfg.get("eval", {}).get("threshold", 0.7)
    max_fpr = cfg.get("eval", {}).get("max_fpr", 0.01)

    f1_metrics = compute_f1_at_threshold(sim_matrix, query_ids, gallery_ids, threshold)
    optimal = find_optimal_threshold(sim_matrix, query_ids, gallery_ids, max_fpr=max_fpr)
    separation = compute_separation(sim_matrix, query_ids, gallery_ids)

    metrics = {
        "rank1": compute_rankk_accuracy(sim_matrix, query_ids, gallery_ids, k=1),
        "rank5": compute_rankk_accuracy(sim_matrix, query_ids, gallery_ids, k=5),
        "rank10": compute_rankk_accuracy(sim_matrix, query_ids, gallery_ids, k=10),
        "map": compute_map(sim_matrix, query_ids, gallery_ids),
        "threshold": threshold,
        "precision": f1_metrics["precision"],
        "recall": f1_metrics["recall"],
        "f1": f1_metrics["f1"],
        "fpr": f1_metrics["fpr"],
        "fnr": f1_metrics["fnr"],
        "tp": f1_metrics["tp"],
        "fp": f1_metrics["fp"],
        "fn": f1_metrics["fn"],
        "tn": f1_metrics["tn"],
        "optimal_threshold": optimal,
        "confidence_margin": compute_confidence_margin(sim_matrix, query_ids, gallery_ids),
        "intra_class": separation["intra_class"],
        "inter_class": separation["inter_class"],
        "separation_ratio": separation["separation_ratio"],
    }
    metrics["aggregate_score"] = compute_aggregate_score(metrics)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if wandb_run is not None:
        wandb_run.log({
            "rank1": metrics["rank1"],
            "rank5": metrics["rank5"],
            "rank10": metrics["rank10"],
            "map": metrics["map"],
            "f1": metrics["f1"],
            "confidence_margin": metrics["confidence_margin"],
            "separation_ratio": metrics["separation_ratio"],
            "aggregate_score": metrics["aggregate_score"],
        })

    return metrics
