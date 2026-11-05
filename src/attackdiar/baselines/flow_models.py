"""Per-flow classifiers. `make_flow_model(kind, ...)` returns an object with
fit(train_flows, val_flows) and predict_proba(flows) -> (n, n_classes)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.features import class_weights, encode, feature_columns, subsample_per_class
from ..metrics.flow import flow_macro_f1
from ..utils import LOG, has_module, resolve_device


class XGBFlowModel:
    """XGBoost on GPU (falls back to sklearn HistGradientBoosting if xgboost is missing)."""

    def __init__(self, classes: list[str], params: dict, cfg: dict, seed: int):
        self.classes, self.params, self.cfg, self.seed = classes, dict(params), cfg, seed
        self.cols: list[str] = []

    def _xy(self, flows: pd.DataFrame):
        return flows[self.cols].to_numpy(np.float32), encode(flows["attack"], self.classes)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        self.cols = feature_columns(train)
        X, y = self._xy(train)
        keep = subsample_per_class(y, self.cfg["flow_models"]["max_train_per_class"], self.seed)
        X, y = X[keep], y[keep]
        w = class_weights(y, len(self.classes))[y]
        Xv, yv = self._xy(val)
        mv = yv >= 0
        Xv, yv = Xv[mv], yv[mv]
        p = dict(self.params)
        es = p.pop("early_stopping_rounds", 30)
        if has_module("xgboost"):
            import xgboost as xgb

            dev = "cuda" if resolve_device(self.cfg) == "cuda" else "cpu"
            self.model = xgb.XGBClassifier(
                objective="multi:softprob", num_class=len(self.classes), tree_method="hist", device=dev,
                n_jobs=self.cfg["compute"]["n_jobs"], random_state=self.seed, eval_metric="mlogloss",
                early_stopping_rounds=es if len(yv) else None, **p)
            self.model.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)] if len(yv) else None, verbose=False)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            LOG.warning("xgboost not installed -> sklearn HistGradientBoosting (slower, CPU)")
            self.model = HistGradientBoostingClassifier(
                max_iter=p.get("n_estimators", 200), learning_rate=p.get("learning_rate", 0.1),
                max_depth=p.get("max_depth", 8), early_stopping=True, random_state=self.seed)
            self.model.fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, flows: pd.DataFrame) -> np.ndarray:
        X = flows[self.cols].to_numpy(np.float32)
        P = self.model.predict_proba(X)
        if P.shape[1] != len(self.classes):  # sklearn: columns only for seen classes
            full = np.zeros((len(X), len(self.classes)), np.float32)
            full[:, self.model.classes_] = P
            P = full
        return P.astype(np.float32)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"classes": self.classes, "cols": self.cols, "model": self.model}, f)

    @classmethod
    def load(cls, path: Path, cfg: dict):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["classes"], {}, cfg, 0)
        obj.cols, obj.model = d["cols"], d["model"]
        return obj


def make_flow_model(kind: str, classes: list[str], cfg: dict, seed: int, params: dict | None = None):
    params = params or cfg["flow_models"][kind]
    if kind == "xgb":
        return XGBFlowModel(classes, params, cfg, seed)
    from .torch_flow import TorchFlowModel

    return TorchFlowModel(kind, classes, params, cfg, seed)


def suggest_params(kind: str, trial, base: dict) -> dict:
    p = dict(base)
    if kind == "xgb":
        p.update(max_depth=trial.suggest_int("max_depth", 4, 12),
                 learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0))
    else:
        p.update(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                 dropout=trial.suggest_float("dropout", 0.0, 0.3))
        if "d_model" in base:
            p["d_model"] = trial.suggest_categorical("d_model", sorted({base["d_model"] // 2, base["d_model"],
                                                                        base["d_model"] * 2}))
    return p


def tune_flow_model(kind, classes, train, val, cfg, seed) -> dict:
    """Optuna search (same n_trials for every model). Objective: val flow macro-F1."""
    base = cfg["flow_models"][kind]
    tcfg = cfg["tuning"]
    if not tcfg["enabled"] or not has_module("optuna"):
        return base
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yv = encode(val["attack"], classes)

    def objective(trial):
        p = suggest_params(kind, trial, base)
        m = make_flow_model(kind, classes, cfg, seed, p).fit(train, val)
        return flow_macro_f1(yv, m.predict_proba(val).argmax(1))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=tcfg["n_trials"], timeout=tcfg["timeout_sec"])
    best = suggest_params(kind, optuna.trial.FixedTrial(study.best_params), base)
    LOG.info("tuned %s: %s (val flow F1 %.4f)", kind, study.best_params, study.best_value)
    return best
