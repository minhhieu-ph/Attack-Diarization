"""Per-flow feature matrices and label encoding for flow classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd

BENIGN = "Benign"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


class Standardizer:
    """log1p on non-negative columns, then z-score. Fit on train only."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = X.astype(np.float64)
        self.log = (X.min(axis=0) >= 0)
        Z = np.where(self.log, np.log1p(np.clip(X, 0, None)), X)
        self.mean = Z.mean(axis=0)
        self.std = Z.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        Z = np.where(self.log, np.log1p(np.clip(X, 0, None)), X)
        return np.nan_to_num((Z - self.mean) / self.std, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def class_list(train_attacks: pd.Series) -> list[str]:
    names = sorted(set(train_attacks.unique()) - {BENIGN})
    return [BENIGN] + names


def encode(attacks: pd.Series, classes: list[str]) -> np.ndarray:
    """Class index per flow; unseen classes -> -1 (ignored in training)."""
    idx = {c: i for i, c in enumerate(classes)}
    return attacks.map(idx).fillna(-1).astype(np.int64).to_numpy()


def subsample_per_class(y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Row indices keeping at most `cap` rows per class."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        if c < 0:
            continue
        idx = np.flatnonzero(y == c)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep)) if keep else np.empty(0, dtype=np.int64)


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse square-root frequency, normalised to mean 1 over present classes."""
    cnt = np.bincount(y[y >= 0], minlength=n_classes).astype(np.float64)
    w = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0.0)
    present = cnt > 0
    w[present] /= w[present].mean()
    return w


def family_map(families: dict) -> dict:
    m = {}
    for fam, names in families.items():
        for n in names:
            m[str(n).lower()] = fam
    return m


def to_family(attacks: pd.Series, fmap: dict) -> pd.Series:
    low = attacks.str.lower()
    return low.map(fmap).where(attacks != BENIGN, BENIGN).fillna("other")
