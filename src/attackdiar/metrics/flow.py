"""Flow-level reference metrics (reported only for comparison)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def flow_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = y_true >= 0
    if not m.any():
        return float("nan")
    labels = np.unique(y_true[m])
    return float(f1_score(y_true[m], y_pred[m], labels=labels, average="macro", zero_division=0))


def flow_macro_f1_names(true_names: np.ndarray, pred_names: np.ndarray) -> float:
    labels = np.unique(true_names)
    return float(f1_score(true_names, pred_names, labels=labels, average="macro", zero_division=0))
