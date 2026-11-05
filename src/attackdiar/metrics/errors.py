"""E3: where does a model fail? One category per GT episode."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .episode import _arrays, match_episodes
from ..utils import jaccard

CATEGORIES = ["correct", "attribution", "boundary", "fragmented", "merged", "confusion", "missed"]


def categorize(gt: pd.DataFrame, pred: pd.DataFrame, mcfg: dict) -> pd.DataFrame:
    gt = gt.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    collar = int(mcfg.get("collar_sec", 5) * 1000)
    jthr = mcfg.get("attribution_jaccard", 0.5)
    gcls, gs, ge, _, gA, gV = _arrays(gt)
    pcls, ps, pe, _, pA, pV = _arrays(pred)
    pairs = {gi: pj for gi, pj, _ in match_episodes(gt, pred, mcfg.get("iou_threshold", 0.5), collar)}

    def overl(s1, e1, s2, e2):
        return (s2 - collar <= e1 + collar) & (e2 + collar >= s1 - collar)

    # does this GT episode overlap another GT episode in time (any class)?
    gt_overlap = np.zeros(len(gt), bool)
    for i in range(len(gt)):
        m = overl(gs[i], ge[i], gs, ge)
        m[i] = False
        gt_overlap[i] = m.any()

    rows = []
    for i in range(len(gt)):
        same = np.flatnonzero((pcls == gcls[i]) & overl(gs[i], ge[i], ps, pe))
        other = np.flatnonzero((pcls != gcls[i]) & overl(gs[i], ge[i], ps, pe))
        if i in pairs:
            j = pairs[i]
            ok = jaccard(gA[i], pA[j]) >= jthr and jaccard(gV[i], pV[j]) >= jthr
            cat = "correct" if ok else "attribution"
        elif len(same) >= 2:
            cat = "fragmented"
        elif len(same) == 1:
            j = same[0]
            shared = ((gcls == pcls[j]) & overl(ps[j], pe[j], gs, ge)).sum()
            cat = "merged" if shared >= 2 else "boundary"
        elif len(other):
            cat = "confusion"
        else:
            cat = "missed"
        rows.append({"gt_index": i, "attack": gcls[i], "category": cat, "overlaps_other_gt": bool(gt_overlap[i]),
                     "duration_s": (ge[i] - gs[i]) / 1000.0})
    return pd.DataFrame(rows)


def summarize(cat: pd.DataFrame) -> pd.DataFrame:
    if cat.empty:
        return pd.DataFrame()
    t = pd.crosstab(cat["attack"], cat["category"], normalize="index").reindex(columns=CATEGORIES, fill_value=0)
    t["n"] = cat.groupby("attack").size()
    ov = cat[cat["overlaps_other_gt"]]
    t["correct_when_overlapping"] = (ov.groupby("attack")["category"].apply(lambda s: (s == "correct").mean())
                                     .reindex(t.index))
    return t.reset_index()
