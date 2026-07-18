"""Loads config.yaml and derives values that depend on it (e.g. REF_DATE_CUTOFF)."""

import os

import pandas as pd
import yaml


def find_config_path(start_dir=None):
    """Searches upward from start_dir (default: cwd) for config.yaml.

    Notebooks in this repo are run with cwd == repo root, but this walks up
    a few levels so the package also works from a subdirectory (e.g. tests/).
    """
    d = os.path.abspath(start_dir or os.getcwd())
    for _ in range(5):
        candidate = os.path.join(d, "config.yaml")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError("config.yaml not found in cwd or any parent directory")


def load_config(path=None):
    """Loads config.yaml and adds derived fields under a top-level 'derived' key."""
    path = path or find_config_path()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    global_max_date = pd.Timestamp(cfg["labels"]["global_max_date"])
    ltv_window_days = cfg["labels"]["ltv_window_days"]
    ref_date_cutoff = global_max_date - pd.Timedelta(days=ltv_window_days)

    cfg["derived"] = {
        "global_max_date": global_max_date,
        "ref_date_cutoff": ref_date_cutoff,
        "repo_root": os.path.dirname(os.path.abspath(path)),
    }
    return cfg


def abspath(cfg, *parts):
    """Resolves a path relative to the repo root that owns the loaded config."""
    return os.path.join(cfg["derived"]["repo_root"], *parts)
