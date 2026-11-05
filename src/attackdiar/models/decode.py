"""Turn FlowDiar host-bin probabilities into episodes (numpy only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.episodes import EP_COLS, empty_episodes, episodes_from_flow_labels, merge_multi_attacker
from ..data.hostbins import HostBins
from ..utils import SEP


def decode_hysteresis(hb: HostBins, p_cls: np.ndarray, p_role: np.ndarray, classes: list[str],
                      high: float, low: float, gap_fill_bins: int, victim_prob: float,
                      avail_ms: np.ndarray | None = None, jacc: float = 0.5):
    """Attacker segments per (host, class): runs of bins with score >= low that contain
    at least one bin >= high; gaps of <= gap_fill_bins empty bins are bridged.
    score = P(class) * P(role = attacker). Victims are the destinations the attacker
    contacted during the segment whose victim probability is >= victim_prob
    (fallback: top-3 destinations by flow count).
    Returns (episodes, predicted class index per row for attacker rows)."""
    row_pred = np.zeros(len(hb.host), np.int64)
    order = np.lexsort((hb.bin, hb.host))
    att = p_role[:, 1]
    runs = []
    for c in range(1, len(classes)):
        s = p_cls[:, c] * att
        idx = order[s[order] >= low]
        if len(idx) == 0:
            continue
        h, b = hb.host[idx], hb.bin[idx]
        new = np.r_[True, (h[1:] != h[:-1]) | ((b[1:] - b[:-1]) > gap_fill_bins + 1)]
        run = np.cumsum(new) - 1
        hi = s[idx] >= high
        keep = np.bincount(run, weights=hi.astype(float)) > 0
        if not keep.any():
            continue
        sel = keep[run]
        idx, run, h, b, hi = idx[sel], run[sel], h[sel], b[sel], hi[sel]
        row_pred[idx] = c
        d = pd.DataFrame({"run": run, "host": h, "bin": b, "row": idx, "hi": hi})
        g = d.groupby("run")
        first_hi = d[d["hi"]].groupby("run")["row"].first()
        r = pd.DataFrame({"host": g["host"].first(), "bs": g["bin"].min(), "be": g["bin"].max(),
                          "first_hi": first_hi}).reset_index(drop=True)
        r["cls"] = c
        runs.append(r)
    if not runs:
        return empty_episodes(), row_pred
    R = pd.concat(runs, ignore_index=True)
    R["rid"] = np.arange(len(R))
    if avail_ms is not None:
        R["t_alert"] = avail_ms[R["first_hi"].to_numpy()]
    else:
        R["t_alert"] = hb.t0 + R["bs"].to_numpy() * hb.bin_ms

    # victims
    E = pd.DataFrame({"src": hb.e_src, "dst": hb.e_dst, "bin": hb.e_bin, "cnt": hb.e_cnt})
    E = E[np.isin(E["src"].to_numpy(), R["host"].unique())]
    M = E.merge(R[["rid", "host", "bs", "be"]], left_on="src", right_on="host")
    M = M[(M["bin"] >= M["bs"]) & (M["bin"] <= M["be"])]
    lookup = hb.row_lookup()
    key = M["dst"].to_numpy(np.int64) * (hb.n_bins + 1) + M["bin"].to_numpy(np.int64)
    rows = lookup.reindex(key).fillna(-1).astype(np.int64).to_numpy()
    M = M.assign(pv=np.where(rows >= 0, p_role[np.clip(rows, 0, None), 2], 0.0))
    V = M.groupby(["rid", "dst"]).agg(pv=("pv", "mean"), cnt=("cnt", "sum")).reset_index()
    vic, nflows = {}, M.groupby("rid")["cnt"].sum()
    for rid, sub in V.groupby("rid"):
        good = sub[sub["pv"] >= victim_prob]
        if good.empty:
            good = sub.nlargest(3, "cnt")
        vic[rid] = SEP.join(sorted(hb.hosts[good["dst"].to_numpy()].astype(str)))

    eps = pd.DataFrame({
        "episode_id": R["rid"].to_numpy(),
        "attack": np.asarray(classes, dtype=object)[R["cls"].to_numpy()],
        "t_start": hb.t0 + R["bs"].to_numpy() * hb.bin_ms,
        "t_end": hb.t0 + (R["be"].to_numpy() + 1) * hb.bin_ms - 1,
        "attackers": hb.hosts[R["host"].to_numpy()].astype(str),
        "victims": [vic.get(i, "") for i in R["rid"]],
        "n_flows": nflows.reindex(R["rid"]).fillna(0).astype(np.int64).to_numpy(),
        "t_alert": R["t_alert"].to_numpy(np.int64),
    })[EP_COLS]
    eps, _ = merge_multi_attacker(eps, eps["episode_id"].to_numpy(), jacc)
    return eps, row_pred


def flow_labels_from_rows(hb: HostBins, row_cls: np.ndarray, classes: list[str]) -> np.ndarray:
    fr = hb.flow_rows()
    c = np.where(fr >= 0, row_cls[np.clip(fr, 0, None)], 0)
    return np.asarray(classes, dtype=object)[c]


def decode_sessionize(hb: HostBins, flows: pd.DataFrame, p_cls, p_role, classes, tau_sec, jacc=0.5):
    """Ablation decoder: per-row argmax (attacker prob >= 0.5) -> flow labels -> B1 grouping."""
    row_cls = np.where(p_role[:, 1] >= 0.5, p_cls[:, 1:].argmax(1) + 1, 0)
    labels = flow_labels_from_rows(hb, row_cls, classes)
    eps, _ = episodes_from_flow_labels(flows, labels, tau_sec * 1000, jacc)
    return eps, row_cls
