import os
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_shared_config(config_dir: str = "configs") -> dict:
    """Load the shared config file if it exists."""
    shared_path = os.path.join(config_dir, "shared.yaml")
    if os.path.exists(shared_path):
        return load_config(shared_path)
    return {}
