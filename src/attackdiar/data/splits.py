"""Train/val/test splits shared by every model.

temporal : cut the time axis at quantiles (moved into an attack-free gap when one
           exists nearby); episodes crossing a cut are clipped per part.
host     : attacker IPs are partitioned; an episode goes to the part of its first
           attacker; benign flows are split by time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import LOG, to_set

PARTS = ("train", "val", "test")


def _union_intervals(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if len(starts) == 0:
        return np.empty((0, 2), dtype=np.int64)
    o = np.argsort(starts)
    s, e = starts[o], ends[o]
    out = [[s[0], e[0]]]
    for a, b in zip(s[1:], e[1:]):
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.asarray(out, dtype=np.int64)


def _move_cut_to_gap(cut: int, busy: np.ndarray, max_shift: int) -> int:
    """If `cut` lies inside a busy interval, move it to the nearest edge within max_shift."""
    for s, e in busy:
        if s <= cut <= e:
            cands = [(cut - (s - 1), s - 1), ((e + 1) - cut, e + 1)]
            d, c = min(cands)
            return int(c) if d <= max_shift else cut
    return cut


def temporal_parts(flows: pd.DataFrame, eps: pd.DataFrame, fractions, gap_search_frac: float,
                   by: str = "flows") -> tuple[np.ndarray, dict]:
    """by='episodes' cuts at quantiles of episode start times so every part gets attacks
    even when the attacks sit in a few days of a long capture (e.g. CIDDS-001)."""
    t = flows["t_start"].to_numpy()
    q = np.cumsum(fractions)[:-1]
    ref = eps["t_start"].to_numpy() if by == "episodes" and len(eps) >= 3 else t
    cuts = [int(np.quantile(ref, x)) for x in q]
    busy = _union_intervals(eps["t_start"].to_numpy(), eps["t_end"].to_numpy())
    span = int(t.max() - t.min()) if len(t) else 0
    cuts = [_move_cut_to_gap(c, busy, int(span * gap_search_frac)) for c in cuts]
    part = np.searchsorted(np.asarray(cuts), t, side="right").astype(np.int8)
    return part, {"cuts_ms": cuts}


def host_parts(flows: pd.DataFrame, eps: pd.DataFrame, fractions, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    first_att = eps["attackers"].map(lambda s: sorted(to_set(s))[0] if to_set(s) else "")
    attackers = np.array(sorted(set(first_att)))
    rng.shuffle(attackers)
    counts = first_att.value_counts()
    # greedy fill parts to the target share of episodes
    target = np.asarray(fractions) * len(eps)
    load = np.zeros(3)
    att_part = {}
    for a in sorted(attackers, key=lambda x: -counts.get(x, 0)):
        p = int(np.argmax(target - load))
        att_part[a] = p
        load[p] += counts.get(a, 0)
    ep_part = first_att.map(att_part).fillna(0).astype(np.int8).to_numpy()
    part, _ = temporal_parts(flows, eps.iloc[:0], fractions, 0.0)  # benign by time
    fe = flows["episode_id"].to_numpy()
    m = fe >= 0
    lookup = pd.Series(ep_part, index=eps["episode_id"].to_numpy())
    part[m] = lookup.reindex(fe[m]).to_numpy()
    info = {"n_attackers": len(attackers),
            "attackers_per_part": [sum(1 for v in att_part.values() if v == k) for k in range(3)]}
    per_class = eps.assign(part=ep_part).groupby("attack")["part"].nunique()
    single = per_class[per_class < 2].index.tolist()
    if single:
        LOG.warning("host split: classes %s have attackers in only one part", single)
    info["classes_single_part"] = single
    return part, info


def make_splits(flows: pd.DataFrame, eps: pd.DataFrame, cfg: dict) -> dict:
    s = cfg["splits"]
    out = {}
    for name in s["names"]:
        if name == "temporal":
            part, info = temporal_parts(flows, eps, s["fractions"], s["gap_search_frac"], s.get("temporal_by", "flows"))
        elif name == "host":
            if len(eps) < 3:
                LOG.warning("host split skipped: <3 episodes")
                continue
            part, info = host_parts(flows, eps, s["fractions"], s["seed"])
        else:
            raise ValueError(f"unknown split {name}")
        out[name] = {"part": part, "info": info}
    return out
