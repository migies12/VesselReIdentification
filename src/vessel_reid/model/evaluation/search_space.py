import copy


def define_search_space(trial) -> dict:
    """
    all hyper parametres that will be varied
    """
    return {
        "train.lr":               trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        "train.triplet.margin":   trial.suggest_float("triplet_margin", 0.1, 0.5),
        "train.arcface_margin":   trial.suggest_float("arcface_margin", 0.3, 0.7),
        "train.arcface_scale":    trial.suggest_categorical("arcface_scale", [15, 30, 64]),
        "train.loss":             trial.suggest_categorical("loss", ["triplet", "combined"]),
        "data.pk_sampler.p":      trial.suggest_categorical("p", [4, 8, 16]),
        "data.pk_sampler.k":      trial.suggest_categorical("k", [4, 8]),
        "data.crop":              trial.suggest_categorical("crop", [True, False]),
        "data.normalize":         trial.suggest_categorical("normalize", [True, False]),
        "model.use_length":       trial.suggest_categorical("use_length", [True, False]),
    }


def apply_overrides(base_cfg: dict, overrides: dict) -> dict:
    """
    converts dot notation to config to write to config in order to vary stuff
    e.g. "train.lr": 0.001 sets cfg["train"]["lr"] = 0.001
    """
    cfg = copy.deepcopy(base_cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg
