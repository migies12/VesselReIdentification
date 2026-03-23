import copy


def apply_overrides(base_cfg: dict, overrides: dict) -> dict:
    """
    applies dot-notation overrides to a deep copy of base_cfg.
    e.g. {"train.lr": 0.001} sets cfg["train"]["lr"] = 0.001.
    base_cfg is never mutated.
    """
    cfg = copy.deepcopy(base_cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg
