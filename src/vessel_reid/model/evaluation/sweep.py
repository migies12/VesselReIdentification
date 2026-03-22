import argparse
import csv
import os
from pathlib import Path

import optuna
import yaml

from vessel_reid.model.evaluation.evaluate import evaluate
from vessel_reid.model.evaluation.search_space import apply_overrides, define_search_space
from vessel_reid.model.inference.build_gallery import build_gallery
from vessel_reid.model.training.training import train
from vessel_reid.model.utils.config import load_config
from vessel_reid import paths


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


def run_trial(trial, base_cfg: dict, runs_dir: Path, use_wandb: bool) -> float:
    """
    one optuna trial: sample params -> train -> build gallery -> evaluate.
    logs to wandb if enabled, appends row to results_summary.csv.
    returns aggregate score for optuna to optimise.
    """
    overrides = define_search_space(trial)
    cfg = apply_overrides(base_cfg, overrides)
    _inject_paths(cfg)

    # use fewer epochs during sweep if sweep_epochs is set in config
    sweep_epochs = cfg.get("sweep", {}).get("sweep_epochs")
    if sweep_epochs is not None:
        cfg["train"]["epochs"] = sweep_epochs

    run_dir = runs_dir / f"run_{trial.number:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="vessel-reid",
            config=cfg,
            name=f"run_{trial.number:04d}",
            reinit=True,
        )

    train(cfg, run_dir, wandb_run=wandb_run)
    build_gallery(cfg, run_dir)
    metrics = evaluate(cfg, run_dir, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    row = {f"param_{k}": v for k, v in overrides.items()}
    row.update({f"metric_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
    row["trial"] = trial.number
    _append_results(runs_dir / "results_summary.csv", row)

    return metrics["aggregate_score"]


def _no_improvement_callback(patience: int):
    """
    stops the study if best score hasn't improved in the last `patience` trials.
    avoids running more trials when the model has plateaued.
    """
    def callback(study, trial):
        if trial.number < patience:
            return
        recent = [t.value for t in study.trials[-patience:] if t.value is not None]
        if recent and max(recent) <= study.best_value - 1e-4:
            study.stop()
    return callback


def run_sweep(
    base_config_path: str,
    runs_dir: Path,
    n_trials: int,
    use_wandb: bool,
    patience: int = 10,
) -> None:
    """
    runs up to n_trials optuna trials, maximising aggregate score.
    stops early if best score hasn't improved in `patience` trials.
    prints best trial params + score at the end.
    """
    base_cfg = load_config(base_config_path)
    runs_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: run_trial(trial, base_cfg, runs_dir, use_wandb),
        n_trials=n_trials,
        callbacks=[_no_improvement_callback(patience)],
    )

    print(f"\nbest trial: #{study.best_trial.number}")
    print(f"best score: {study.best_value:.4f}")
    print("best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run hyperparameter sweep")
    parser.add_argument("--config", required=True, help="path to base config YAML")
    parser.add_argument("--runs-dir", default="runs", help="directory to write all run outputs")
    parser.add_argument("--n-trials", type=int, default=30, help="max number of optuna trials")
    parser.add_argument("--patience", type=int, default=10, help="stop after this many trials with no improvement")
    parser.add_argument("--wandb", action="store_true", help="enable W&B logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(
        base_config_path=args.config,
        runs_dir=Path(args.runs_dir),
        n_trials=args.n_trials,
        use_wandb=args.wandb,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
