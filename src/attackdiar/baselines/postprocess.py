"""Turn per-flow predictions into episodes (post-processing of B1-B7).

Every function returns (episodes, episode id per flow) and uses the same episode
schema as the ground truth. Parameters are tuned on the validation part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.episodes import (episodes_from_flow_labels, empty_episodes, merge_multi_attacker,
                             segments_to_episodes, sessionize)
from ..data.features import BENIGN
from ..metrics.episode import episode_metrics


def _labels(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.asarray(classes, dtype=object)[proba.argmax(1)]


# ---------------------------------------------------------------- B1 / B4-B6
def post_sessionize(flows, proba, classes, tau_sec, jacc=0.5):
    return episodes_from_flow_labels(flows, _labels(proba, classes), tau_sec * 1000, jacc)


# ---------------------------------------------------------------- B2: HMM smoothing
def viterbi_per_host(flows: pd.DataFrame, proba: np.ndarray, p_stay: float) -> np.ndarray:
    """Sticky-HMM Viterbi over each source host's flow sequence (time order).
    Emissions = classifier posteriors. Only hosts with >= 1 predicted attack are
    smoothed. Returns smoothed class index per flow."""
    C = proba.shape[1]
    pred = proba.argmax(1)
    out = pred.copy()
    src = flows["src_ip"].to_numpy()
    hosts = np.unique(src[pred != 0])
    if len(hosts) == 0 or C < 2:
        return out
    idx = np.flatnonzero(np.isin(src, hosts))
    codes, _ = pd.factorize(src[idx])
    order = np.lexsort((flows["t_start"].to_numpy()[idx], codes))
    idx, codes = idx[order], codes[order]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    lengths = np.diff(np.r_[starts, len(idx)])
    g_order = np.argsort(-lengths, kind="stable")  # longest first -> active groups are a prefix
    starts, lengths = starts[g_order], lengths[g_order]
    logA = np.full((C, C), np.log((1 - p_stay) / (C - 1)))
    np.fill_diagonal(logA, np.log(p_stay))
    logE = np.log(proba[idx] + 1e-9)
    back = np.zeros((len(idx), C), dtype=np.int16)
    delta = logE[starts].copy()
    neg_len = -lengths
    for k in range(1, int(lengths.max())):
        na = np.searchsorted(neg_len, -k, side="left")  # groups with length > k
        e = starts[:na] + k
        sc = delta[:na, :, None] + logA[None]
        arg = sc.argmax(1)
        back[e] = arg
        delta[:na] = np.take_along_axis(sc, arg[:, None, :], 1)[:, 0, :] + logE[e]
    state = delta.argmax(1)
    path = np.empty(len(idx), dtype=np.int64)
    last = starts + lengths - 1
    path[last] = state
    for k in range(int(lengths.max()) - 1, 0, -1):
        na = np.searchsorted(neg_len, -k, side="left")
        e = starts[:na] + k
        state[:na] = back[e, state[:na]]
        path[e - 1] = state[:na]
    out[idx] = path
    return out


def post_hmm(flows, proba, classes, p_stay, tau_sec, jacc=0.5):
    sm = viterbi_per_host(flows, proba, p_stay)
    return episodes_from_flow_labels(flows, np.asarray(classes, dtype=object)[sm], tau_sec * 1000, jacc)


# ---------------------------------------------------------------- B3: DBSCAN on alerts
def dbscan_1d(t: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Exact DBSCAN on sorted 1-D points in O(n log n). Returns cluster id or -1."""
    n = len(t)
    lab = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return lab
    cnt = np.searchsorted(t, t + eps, "right") - np.searchsorted(t, t - eps, "left")
    core = np.flatnonzero(cnt >= min_samples)
    if len(core) == 0:
        return lab
    ct = t[core]
    cid = np.cumsum(np.r_[True, np.diff(ct) > eps]) - 1
    lab[core] = cid
    border = np.flatnonzero(cnt < min_samples)
    if len(border):
        k = np.searchsorted(ct, t[border])
        left = np.clip(k - 1, 0, len(ct) - 1)
        right = np.clip(k, 0, len(ct) - 1)
        dl, dr = np.abs(t[border] - ct[left]), np.abs(ct[right] - t[border])
        nearest = np.where(dl <= dr, left, right)
        ok = np.minimum(dl, dr) <= eps
        lab[border[ok]] = cid[nearest[ok]]
    return lab


def post_dbscan(flows, proba, classes, eps_sec, min_samples, jacc=0.5):
    labels = _labels(proba, classes)
    flow_ep = np.full(len(flows), -1, dtype=np.int64)
    t = flows["t_start"].to_numpy()
    seg = np.full(len(flows), -1, dtype=np.int64)
    nxt = 0
    for c in np.unique(labels):
        if c == BENIGN:
            continue
        idx = np.flatnonzero(labels == c)  # flows are time-sorted
        lab = dbscan_1d(t[idx].astype(np.float64), eps_sec * 1000.0, min_samples)
        ok = lab >= 0
        seg[idx[ok]] = lab[ok] + nxt
        nxt += int(lab.max()) + 1 if ok.any() else 0
    m = seg >= 0
    if not m.any():
        return empty_episodes(), flow_ep
    f = flows.loc[m]
    eps = segments_to_episodes(f["t_start"].to_numpy(), f["t_end"].to_numpy(), f["src_ip"].to_numpy(),
                               f["dst_ip"].to_numpy(), labels[m], seg[m])
    eps, s2 = merge_multi_attacker(eps, seg[m], jacc)
    flow_ep[m] = s2
    return eps, flow_ep


# ---------------------------------------------------------------- B7: SAGE-style episodes
def _valley_split(ts: np.ndarray, bin_ms: int, ratio: float) -> np.ndarray:
    """Split one alert burst at frequency valleys (SAGE: an episode spans a rise and
    fall of alert frequency). Returns sub-segment id per alert."""
    b = (ts - ts.min()) // bin_ms
    counts = np.bincount(b)
    sm = np.convolve(counts, np.ones(3) / 3, mode="same")
    cut_bins, peak = [], 0.0
    for i in range(1, len(sm) - 1):
        peak = max(peak, sm[i - 1])
        if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1] and sm[i] <= ratio * peak:
            cut_bins.append(i)
            peak = 0.0
    return np.searchsorted(np.asarray(cut_bins), b, side="right")


def post_sage(flows, proba, classes, bin_sec, tau_sec, valley_ratio, min_alerts_split, jacc=0.5):
    labels = _labels(proba, classes)
    flow_ep = np.full(len(flows), -1, dtype=np.int64)
    m = labels != BENIGN
    if not m.any():
        return empty_episodes(), flow_ep
    f = flows.loc[m]
    lab = labels[m]
    ts, te = f["t_start"].to_numpy(), f["t_end"].to_numpy()
    src, dst = f["src_ip"].to_numpy(), f["dst_ip"].to_numpy()
    pair = np.char.add(np.char.add(src.astype(str), "|"), dst.astype(str))
    _, seg = sessionize(ts, te, pair, dst, lab, tau_sec * 1000)  # per (attacker, victim, class)
    # split long pair-episodes at frequency valleys
    seg = seg.copy()
    nxt = seg.max() + 1
    sizes = np.bincount(seg)
    for s in np.flatnonzero(sizes >= min_alerts_split):
        idx = np.flatnonzero(seg == s)
        sub = _valley_split(ts[idx], int(bin_sec * 1000), valley_ratio)
        if sub.max() > 0:
            seg[idx[sub > 0]] = nxt + sub[sub > 0] - 1
            nxt += int(sub.max())
    # attacker-level episodes: union of overlapping pair-episodes of the same (src, class)
    d = pd.DataFrame({"seg": seg, "src": src, "cls": lab, "ts": ts, "te": te})
    ps = d.groupby("seg").agg(src=("src", "first"), cls=("cls", "first"), ts=("ts", "min"), te=("te", "max"))
    ps = ps.sort_values(["src", "cls", "ts"])
    newgrp = (ps["src"] != ps["src"].shift()) | (ps["cls"] != ps["cls"].shift())
    gid = newgrp.cumsum()
    runmax = ps.groupby(gid)["te"].cummax()
    prev = runmax.groupby(gid).shift()
    newep = newgrp | (ps["ts"] > prev)
    ep_of_seg = pd.Series((newep.cumsum() - 1).to_numpy(), index=ps.index)
    ep = ep_of_seg.reindex(seg).to_numpy()
    eps = segments_to_episodes(ts, te, src, dst, lab, ep)
    eps, ep2 = merge_multi_attacker(eps, ep, jacc)
    flow_ep[np.flatnonzero(m)] = ep2
    return eps, flow_ep


# ---------------------------------------------------------------- tuning
def candidates(kind: str, pcfg: dict) -> list[dict]:
    c = pcfg[kind]
    if kind == "sessionize":
        return [{"tau_sec": t} for t in c["tau_grid_sec"]]
    if kind == "hmm":
        return [{"p_stay": p, "tau_sec": t} for p in c["p_stay_grid"] for t in c["tau_grid_sec"]]
    if kind == "dbscan":
        return [{"eps_sec": e, "min_samples": c["min_samples"]} for e in c["eps_grid_sec"]]
    if kind == "sage":
        return [{"bin_sec": c["bin_sec"], "tau_sec": t, "valley_ratio": c["valley_ratio"],
                 "min_alerts_split": c["min_alerts_split"]} for t in c["tau_grid_sec"]]
    raise ValueError(kind)


FUNCS = {"sessionize": post_sessionize, "hmm": post_hmm, "dbscan": post_dbscan, "sage": post_sage}


def apply_post(kind, flows, proba, classes, params, jacc=0.5):
    return FUNCS[kind](flows, proba, classes, **params, jacc=jacc)


def tune_post(kind, flows_val, proba_val, gt_val, classes, cfg) -> tuple[dict, float]:
    best, best_f1 = None, -1.0
    jacc = cfg["episodes"]["merge_victim_jaccard"]
    for params in candidates(kind, cfg["post"]):
        eps, _ = apply_post(kind, flows_val, proba_val, classes, params, jacc)
        f1 = episode_metrics(gt_val, eps, cfg["metrics"])["f1_macro"]
        if f1 > best_f1:
            best, best_f1 = params, f1
    return best, best_f1
