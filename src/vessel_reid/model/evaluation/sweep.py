import argparse
import csv
from pathlib import Path

import wandb
import yaml

from vessel_reid.model.evaluation.evaluate import evaluate
from vessel_reid.model.evaluation.search_space import apply_overrides
from vessel_reid.model.inference.build_gallery import build_gallery
from vessel_reid.model.training.training import train
from vessel_reid.model.utils.config import load_config
from vessel_reid import paths


# maps W&B sweep parameter names to nested config dot-paths
PARAM_TO_CFG = {
    "lr":             "train.lr",
    "triplet_margin": "train.triplet.margin",
    "arcface_margin": "train.arcface_margin",
    "arcface_scale":  "train.arcface_scale",
    "loss":           "train.loss",
    "p":              "data.pk_sampler.p",
    "k":              "data.pk_sampler.k",
    "crop":           "data.crop",
    "normalize":      "data.normalize",
    "use_length":     "model.use_length",
}


def _inject_paths(cfg: dict) -> None:
    """fill in data/gallery/query paths from paths.py if not already in cfg"""
    cfg["data"].setdefault("csv_path", str(paths.TRAIN_CSV))
    cfg["data"].setdefault("image_root", str(paths.RAW_IMAGES_DIR))
    cfg["gallery"].setdefault("csv_path", str(paths.GALLERY_CSV))
    cfg["gallery"].setdefault("image_root", str(paths.RAW_IMAGES_DIR))
    cfg.setdefault("query", {})
    cfg["query"].setdefault("csv_path", str(paths.QUERY_CSV))
    cfg["query"].setdefault("image_root", str(paths.RAW_IMAGES_DIR))


def _append_results(csv_path: Path, row: dict) -> None:
    """add row to summary csv"""
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_trial_fn(base_cfg: dict, runs_dir: Path):
    """
    returns the function passed to wandb.agent.
    uses a closure so base_cfg and runs_dir are available without globals.
    each call corresponds to one sweep trial.
    """
    trial_counter = [0]

    def trial_fn():
        run = wandb.init()
        trial_num = trial_counter[0]
        trial_counter[0] += 1

        # map flat W&B params back to nested config keys
        overrides = {PARAM_TO_CFG[k]: v for k, v in dict(run.config).items() if k in PARAM_TO_CFG}
        cfg = apply_overrides(base_cfg, overrides)
        _inject_paths(cfg)

        # use fewer epochs during sweep if sweep_epochs is set in base config
        sweep_epochs = cfg.get("sweep", {}).get("sweep_epochs")
        if sweep_epochs is not None:
            cfg["train"]["epochs"] = sweep_epochs

        run_dir = runs_dir / f"run_{trial_num:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f)

        train(cfg, run_dir, wandb_run=run)
        build_gallery(cfg, run_dir)
        metrics = evaluate(cfg, run_dir, wandb_run=run)

        # log the aggregate score so W&B sweep can optimise it
        wandb.log({"aggregate_score": metrics["aggregate_score"]})

        row = {f"param_{k}": v for k, v in overrides.items()}
        row.update({f"metric_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        row["trial"] = trial_num
        _append_results(runs_dir / "results_summary.csv", row)

        run.finish()

    return trial_fn


def run_sweep(
    base_config_path: str,
    sweep_config_path: str,
    runs_dir: Path,
    n_trials: int,
) -> None:
    """
    creates a W&B sweep from sweep_config_path and runs n_trials trials.
    W&B handles Bayesian parameter selection and tracks all results.
    """
    base_cfg = load_config(base_config_path)
    runs_dir.mkdir(parents=True, exist_ok=True)

    with open(sweep_config_path) as f:
        sweep_cfg = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_cfg, project="vessel-reid")
    trial_fn = make_trial_fn(base_cfg, runs_dir)
    wandb.agent(sweep_id, trial_fn, count=n_trials)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run W&B hyperparameter sweep")
    parser.add_argument("--config", required=True, help="path to base model config YAML")
    parser.add_argument("--sweep-config", required=True, help="path to W&B sweep config YAML")
    parser.add_argument("--runs-dir", default="runs", help="directory to write all run outputs")
    parser.add_argument("--n-trials", type=int, default=40, help="max number of trials")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(
        base_config_path=args.config,
        sweep_config_path=args.sweep_config,
        runs_dir=Path(args.runs_dir),
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
