"""Significance tests and confidence intervals."""
from __future__ import annotations

import itertools

import numpy as np


def paired_permutation_test(a, b, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip test on paired differences (e.g. per-seed scores).
    Exact enumeration when n <= 16."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    if n <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, n))
    null = np.abs((signs * d).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def bootstrap_f1_ci(details: dict, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """CI for macro episode F1 by resampling GT and predicted episodes within class.
    Approximation: matched flags are kept fixed per episode."""
    rng = np.random.default_rng(seed)
    gcls, pcls = details["gcls"], details["pcls"]
    gm, pm = details["g_match"] >= 0, details["p_match"] >= 0
    classes = sorted(set(gcls) | set(pcls))
    vals = []
    for _ in range(n_boot):
        f1s = []
        for c in classes:
            g = gm[gcls == c]
            p = pm[pcls == c]
            r = rng.choice(g, len(g)).mean() if len(g) else 0.0
            pr = rng.choice(p, len(p)).mean() if len(p) else 0.0
            f1s.append(2 * pr * r / (pr + r) if pr + r > 0 else 0.0)
        vals.append(np.mean(f1s))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
