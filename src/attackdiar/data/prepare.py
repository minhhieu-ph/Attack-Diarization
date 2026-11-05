"""raw files -> dataset/processed/<name>/{flows, episodes, splits.npz, meta.json}."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import dataset_dir
from ..utils import LOG, check_memory, df_exists, load_df, load_json, save_df, save_json, timer
from .episodes import build_ground_truth, restrict_episodes
from .loaders import load_dataset_raw
from .splits import PARTS, make_splits


def processed_dir(cfg: dict, name: str) -> Path:
    return dataset_dir(cfg) / "processed" / name


def prepare_dataset(cfg: dict, name: str, force: bool = False) -> None:
    out = processed_dir(cfg, name)
    if df_exists(out / "flows") and (out / "meta.json").exists() and not force:
        LOG.info("[%s] already prepared -> skip (use --force to rebuild)", name)
        return
    raw = dataset_dir(cfg) / "raw" / name
    with timer(f"[{name}] load raw"):
        flows = load_dataset_raw(cfg, name, raw)
    LOG.info("[%s] %d flows, %d attack flows", name, len(flows), int(flows["label"].sum()))
    check_memory(cfg, f"{name} loaded")
    with timer(f"[{name}] ground truth"):
        eps, flow_ep, meta = build_ground_truth(flows, cfg, name, dataset_dir(cfg))
    flows["episode_id"] = flow_ep
    with timer(f"[{name}] splits"):
        splits = make_splits(flows, eps, cfg)

    meta.update({
        "dataset": name, "n_flows": len(flows), "n_attack_flows": int(flows["label"].sum()),
        "t_min": int(flows["t_start"].min()), "t_max": int(flows["t_end"].max()),
        "classes": sorted(set(flows["attack"].unique())),
        "episode_classes": eps["attack"].value_counts().to_dict(),
        "splits": {k: v["info"] for k, v in splits.items()},
    })
    out.mkdir(parents=True, exist_ok=True)
    save_df(flows, out / "flows")
    save_df(eps, out / "episodes")
    np.savez_compressed(out / "splits.npz", **{k: v["part"] for k, v in splits.items()})
    save_json(meta, out / "meta.json")
    LOG.info("[%s] saved -> %s (gt=%s, %d episodes)", name, out, meta["gt_source"], len(eps))


class Dataset:
    """Prepared dataset with convenient access to split parts."""

    def __init__(self, cfg: dict, name: str):
        d = processed_dir(cfg, name)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"{name} not prepared: run scripts/prepare.py --datasets {name}")
        self.name = name
        self.meta = load_json(d / "meta.json")
        self.flows = load_df(d / "flows")
        self.episodes = load_df(d / "episodes")
        with np.load(d / "splits.npz") as z:
            self.split_parts = {k: z[k] for k in z.files}

    def splits(self) -> list[str]:
        return list(self.split_parts)

    def part(self, split: str, part: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        p = self.split_parts[split]
        f = self.flows.loc[p == PARTS.index(part)].reset_index(drop=True)
        return f, restrict_episodes(self.episodes, f)
