"""Episode-level metrics.

Matching: within each attack class, GT and predicted episodes are paired one-to-one
(Hungarian) on temporal IoU computed after padding both intervals by `collar`.
A pair counts as a true positive when IoU >= iou_threshold. A long prediction can
therefore match at most one GT episode (no credit for covering several).

Reported (all macro over classes unless stated):
  f1_macro, precision_macro, recall_macro, f1_micro
  attr_acc_macro   share of matched pairs whose attacker AND victim Jaccard >= thr
  attr_recall      share of GT episodes matched with correct attribution (micro)
  att_jaccard, vic_jaccard   mean Jaccard over matched pairs
  eer              DER-style error rate: (miss + false alarm + class confusion) time
                   / GT time, scored outside a +-collar zone around GT boundaries
  ttd_median_s, ttd_p90_s, detect_rate
                   time from GT start to the first alert of an overlapping prediction of
                   the same class sharing an attacker (t_alert, or t_start if absent)
  fa_per_hour      predictions overlapping no GT of their class / benign hours
  alert_compression  alert flows / predicted episodes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..utils import jaccard, to_set


def _arrays(eps: pd.DataFrame):
    cls = eps["attack"].astype(str).to_numpy()
    s = eps["t_start"].to_numpy(dtype=np.int64)
    e = eps["t_end"].to_numpy(dtype=np.int64)
    alert = eps["t_alert"].to_numpy(dtype=np.int64) if "t_alert" in eps.columns else s
    A = [to_set(x) for x in eps["attackers"]]
    V = [to_set(x) for x in eps["victims"]]
    return cls, s, e, alert, A, V


def _overlap_pairs(gs, ge, ps, pe):
    """All (i, j) with [gs_i, ge_i) overlapping [ps_j, pe_j)."""
    order = np.argsort(ps)
    ps_o, pe_o = ps[order], pe[order]
    gi, pj = [], []
    for i in range(len(gs)):
        k = np.searchsorted(ps_o, ge[i], side="left")
        if k == 0:
            continue
        cand = np.flatnonzero(pe_o[:k] > gs[i])
        if len(cand):
            gi.append(np.full(len(cand), i))
            pj.append(order[cand])
    if not gi:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(gi), np.concatenate(pj)


def match_episodes(gt: pd.DataFrame, pred: pd.DataFrame, iou_thr: float = 0.5, collar_ms: int = 0):
    """Return list of (gt_index, pred_index, iou) for matched pairs (positional indices)."""
    gcls, gs, ge, _, _, _ = _arrays(gt)
    pcls, ps, pe, _, _, _ = _arrays(pred)
    gs, ge = gs - collar_ms, ge + collar_ms + 1
    ps, pe = ps - collar_ms, pe + collar_ms + 1
    out = []
    for c in np.intersect1d(np.unique(gcls), np.unique(pcls)):
        gi = np.flatnonzero(gcls == c)
        pj = np.flatnonzero(pcls == c)
        a, b = _overlap_pairs(gs[gi], ge[gi], ps[pj], pe[pj])
        if len(a) == 0:
            continue
        inter = np.minimum(ge[gi][a], pe[pj][b]) - np.maximum(gs[gi][a], ps[pj][b])
        union = (ge[gi][a] - gs[gi][a]) + (pe[pj][b] - ps[pj][b]) - inter
        iou = inter / np.maximum(union, 1)
        keep = iou >= iou_thr
        a, b, iou = a[keep], b[keep], iou[keep]
        if len(a) == 0:
            continue
        n_g, n_p = len(gi), len(pj)
        adj = coo_matrix((np.ones(len(a)), (a, n_g + b)), shape=(n_g + n_p, n_g + n_p))
        _, comp = connected_components(adj, directed=False)
        pair_comp = comp[a]
        for k in np.unique(pair_comp):
            m = pair_comp == k
            ua, ia = np.unique(a[m], return_inverse=True)
            ub, ib = np.unique(b[m], return_inverse=True)
            cost = np.full((len(ua), len(ub)), 1e6)
            cost[ia, ib] = -iou[m]
            r, cidx = linear_sum_assignment(cost)
            for rr, cc in zip(r, cidx):
                if cost[rr, cc] < 0:
                    out.append((int(gi[ua[rr]]), int(pj[ub[cc]]), float(-cost[rr, cc])))
    return out


def _union(starts, ends):
    if len(starts) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    o = np.argsort(starts)
    s, e = starts[o], ends[o]
    us, ue = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a <= ue[-1]:
            ue[-1] = max(ue[-1], b)
        else:
            us.append(a)
            ue.append(b)
    return np.asarray(us), np.asarray(ue)


def _covered(points, us, ue):
    if len(us) == 0:
        return np.zeros(len(points), bool)
    k = np.searchsorted(us, points, side="right") - 1
    return (k >= 0) & (points < ue[np.clip(k, 0, None)])


def episode_error_rate(gt: pd.DataFrame, pred: pd.DataFrame, collar_ms: int) -> float:
    gcls, gs, ge, *_ = _arrays(gt)
    pcls, ps, pe, *_ = _arrays(pred)
    ge, pe = ge + 1, pe + 1
    if len(gs) == 0:
        return float("nan")
    ns_s = np.concatenate([gs - collar_ms, ge - collar_ms])
    ns_e = np.concatenate([gs + collar_ms, ge + collar_ms])
    bps = np.unique(np.concatenate([gs, ge, ps, pe, ns_s, ns_e]))
    a, b = bps[:-1], bps[1:]
    mid = (a + b) / 2.0
    length = (b - a).astype(np.float64)
    scored = ~_covered(mid, *_union(ns_s, ns_e)) if collar_ms > 0 else np.ones(len(mid), bool)
    nref = np.zeros(len(mid))
    nhyp = np.zeros(len(mid))
    ncor = np.zeros(len(mid))
    for c in np.union1d(np.unique(gcls), np.unique(pcls)):
        r = _covered(mid, *_union(gs[gcls == c], ge[gcls == c]))
        h = _covered(mid, *_union(ps[pcls == c], pe[pcls == c]))
        nref += r
        nhyp += h
        ncor += r & h
    L = length * scored
    total = (L * nref).sum()
    if total <= 0:
        return float("nan")
    miss = (L * np.maximum(0, nref - nhyp)).sum()
    fa = (L * np.maximum(0, nhyp - nref)).sum()
    conf = (L * (np.minimum(nref, nhyp) - ncor)).sum()
    return float((miss + fa + conf) / total)


def benign_hours(test_flows: pd.DataFrame, gt: pd.DataFrame, minute_sec: int = 60) -> float:
    """Hours of observed traffic (minutes with >= 1 flow) not overlapping any GT episode."""
    if test_flows is None or len(test_flows) == 0:
        return float("nan")
    step = minute_sec * 1000
    minutes = np.unique(test_flows["t_start"].to_numpy() // step)
    us, ue = _union(gt["t_start"].to_numpy(dtype=np.int64), gt["t_end"].to_numpy(dtype=np.int64) + 1)
    if len(us):
        m_s, m_e = minutes * step, (minutes + 1) * step
        k = np.searchsorted(us, m_e, side="left") - 1
        busy = (k >= 0) & (ue[np.clip(k, 0, None)] > m_s)
        minutes = minutes[~busy]
    return len(minutes) * minute_sec / 3600.0


def episode_metrics(gt: pd.DataFrame, pred: pd.DataFrame, mcfg: dict, test_flows: pd.DataFrame | None = None,
                    n_alert_flows: int | None = None, return_details: bool = False):
    iou_thr = mcfg.get("iou_threshold", 0.5)
    collar = int(mcfg.get("collar_sec", 5) * 1000)
    jthr = mcfg.get("attribution_jaccard", 0.5)
    gt = gt.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    gcls, gs, ge, _, gA, gV = _arrays(gt)
    pcls, ps, pe, palert, pA, pV = _arrays(pred)
    pairs = match_episodes(gt, pred, iou_thr, collar)

    g_match = np.full(len(gt), -1)
    p_match = np.full(len(pred), -1)
    att_j, vic_j, attr_ok = np.zeros(len(gt)), np.zeros(len(gt)), np.zeros(len(gt), bool)
    for gi, pj, _ in pairs:
        g_match[gi], p_match[pj] = pj, gi
        att_j[gi] = jaccard(gA[gi], pA[pj])
        vic_j[gi] = jaccard(gV[gi], pV[pj])
        attr_ok[gi] = att_j[gi] >= jthr and vic_j[gi] >= jthr

    classes = sorted(set(gcls) | set(pcls))
    per_class, f1s, ps_, rs_, attr_c = {}, [], [], [], []
    for c in classes:
        n_g, n_p = int((gcls == c).sum()), int((pcls == c).sum())
        tp = int(((gcls == c) & (g_match >= 0)).sum())
        p = tp / n_p if n_p else 0.0
        r = tp / n_g if n_g else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        m = (gcls == c) & (g_match >= 0)
        a = float(attr_ok[m].mean()) if m.any() else float("nan")
        per_class[c] = {"n_gt": n_g, "n_pred": n_p, "tp": tp, "precision": p, "recall": r, "f1": f, "attr_acc": a}
        f1s.append(f), ps_.append(p), rs_.append(r)
        if n_g:
            attr_c.append(0.0 if np.isnan(a) else a)
    tp_all = int((g_match >= 0).sum())
    P = tp_all / max(1, len(pred))
    R = tp_all / max(1, len(gt))
    matched = g_match >= 0

    # time to detect
    ttd = []
    for i in range(len(gt)):
        m = (pcls == gcls[i]) & (ps - collar <= ge[i] + collar) & (pe + collar >= gs[i] - collar)
        cand = [j for j in np.flatnonzero(m) if gA[i] & pA[j]]
        if cand:
            ttd.append(max(0, int(palert[cand].min()) - int(gs[i])) / 1000.0)
    ttd = np.asarray(ttd)

    # pure false alarms: predictions overlapping no GT of their class
    fa = 0
    for j in range(len(pred)):
        m = (gcls == pcls[j]) & (gs - collar <= pe[j] + collar) & (ge + collar >= ps[j] - collar)
        fa += int(not m.any())
    hours = benign_hours(test_flows, gt, mcfg.get("minute_sec", 60)) if test_flows is not None else float("nan")

    out = {
        "n_gt": len(gt), "n_pred": len(pred), "tp": tp_all,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "precision_macro": float(np.mean(ps_)) if ps_ else 0.0,
        "recall_macro": float(np.mean(rs_)) if rs_ else 0.0,
        "f1_micro": 2 * P * R / (P + R) if P + R > 0 else 0.0,
        "attr_acc_macro": float(np.mean(attr_c)) if attr_c else float("nan"),
        "attr_recall": float(attr_ok.sum() / max(1, len(gt))),
        "att_jaccard": float(att_j[matched].mean()) if matched.any() else float("nan"),
        "vic_jaccard": float(vic_j[matched].mean()) if matched.any() else float("nan"),
        "eer": episode_error_rate(gt, pred, collar),
        "detect_rate": float(len(ttd) / max(1, len(gt))),
        "ttd_median_s": float(np.median(ttd)) if len(ttd) else float("nan"),
        "ttd_p90_s": float(np.percentile(ttd, 90)) if len(ttd) else float("nan"),
        "false_alarm_episodes": fa,
        "benign_hours": hours,
        "fa_per_hour": fa / hours if hours and hours > 0 else float("nan"),
        "alert_compression": (n_alert_flows / max(1, len(pred))) if n_alert_flows is not None else float("nan"),
        "per_class": per_class,
    }
    if return_details:
        return out, {"gcls": gcls, "pcls": pcls, "g_match": g_match, "p_match": p_match, "attr_ok": attr_ok}
    return out
