"""YAML config loading with deep-merge and `key.sub=value` overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def _deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_config(paths: list[str] | None = None, overrides: list[str] | None = None) -> dict:
    """Load configs/default.yaml, then each extra YAML, then `a.b=value` overrides."""
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    for p in paths or []:
        p = Path(p)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        with open(p) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_path(cfg, key.strip(), yaml.safe_load(raw))
    with open(CONFIG_DIR / "datasets.yaml") as f:
        cfg["registry"] = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def dataset_dir(cfg: dict) -> Path:
    p = Path(cfg["project"]["dataset_dir"])
    return p if p.is_absolute() else ROOT / p


def output_dir(cfg: dict) -> Path:
    p = Path(cfg["project"]["output_dir"])
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p
