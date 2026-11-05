"""E7 perturbations of a test part (flows + GT episodes). Models are NOT retrained."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import SEP, to_set
from .episodes import restrict_episodes


def _finish(flows: pd.DataFrame, eps: pd.DataFrame, new_eps: pd.DataFrame | None = None):
    flows = flows.sort_values("t_start", kind="mergesort").reset_index(drop=True)
    base = new_eps if new_eps is not None else eps
    return flows, restrict_episodes(base, flows)


def slow_down(flows: pd.DataFrame, eps: pd.DataFrame, factor: float, seed: int = 0):
    """Stretch every episode's flow times by `factor` around its start (low-and-slow)."""
    f = flows.copy()
    m = f["episode_id"].to_numpy() >= 0
    start = eps.set_index("episode_id")["t_start"]
    s = start.reindex(f.loc[m, "episode_id"]).to_numpy()
    dur = (f.loc[m, "t_end"] - f.loc[m, "t_start"]).to_numpy()
    new_ts = s + ((f.loc[m, "t_start"].to_numpy() - s) * factor).astype(np.int64)
    f.loc[m, "t_start"] = new_ts
    f.loc[m, "t_end"] = new_ts + dur
    return _finish(f, eps)


def interleave_benign(flows: pd.DataFrame, eps: pd.DataFrame, rate: float, seed: int = 0, cap: int = 200_000):
    """Attackers also send benign-looking flows (copied from real benign flows) during
    their episodes: rate x n_flows extra flows per episode."""
    rng = np.random.default_rng(seed)
    ben = flows[flows["label"] == 0]
    if ben.empty:
        return flows, eps
    add = []
    for r in eps.itertuples(index=False):
        k = int(min(cap, rate * r.n_flows))
        A = sorted(to_set(r.attackers))
        if k == 0 or not A:
            continue
        x = ben.iloc[rng.integers(0, len(ben), k)].copy()
        dur = (x["t_end"] - x["t_start"]).to_numpy()
        ts = rng.integers(r.t_start, max(r.t_end, r.t_start + 1), k)
        x["t_start"], x["t_end"] = ts, ts + dur
        x["src_ip"] = rng.choice(A, k)
        x["episode_id"] = -1
        add.append(x)
    if not add:
        return flows, eps
    return _finish(pd.concat([flows, *add], ignore_index=True), eps)


def split_sources(flows: pd.DataFrame, eps: pd.DataFrame, k: int, seed: int = 0):
    """Spread each episode's attacker traffic over k source IPs (round-robin)."""
    f = flows.copy()
    e = eps.copy()
    m = f["episode_id"].to_numpy() >= 0
    src = f.loc[m, "src_ip"].astype(str).to_numpy()
    j = np.arange(m.sum()) % k
    f.loc[m, "src_ip"] = np.char.add(np.char.add(src.astype(str), "#"), j.astype(str))
    e["attackers"] = e["attackers"].map(lambda s: SEP.join(sorted(f"{a}#{i}" for a in to_set(s) for i in range(k))))
    return _finish(f, e, e)


PERTURB = {"slow": slow_down, "interleave": interleave_benign, "split_sources": split_sources}
