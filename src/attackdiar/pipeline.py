"""Train/evaluate one model on one (dataset, split, seed) and return result rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baselines.flow_models import XGBFlowModel, make_flow_model, tune_flow_model
from .baselines.postprocess import apply_post, tune_post
from .config import output_dir
from .data.features import BENIGN, class_list, encode, feature_columns, family_map, to_family
from .data.hostbins import build_hostbins
from .metrics.episode import episode_metrics
from .metrics.flow import flow_macro_f1, flow_macro_f1_names
from .metrics.stats import bootstrap_f1_ci
from .utils import LOG, load_json, save_df, save_json, set_seed, timer

SUMMARY_KEYS = ["f1_macro", "precision_macro", "recall_macro", "f1_micro", "attr_acc_macro", "attr_recall",
                "att_jaccard", "vic_jaccard", "eer", "detect_rate", "ttd_median_s", "ttd_p90_s",
                "fa_per_hour", "alert_compression", "n_gt", "n_pred"]


def model_dir(cfg, ds, split, model, seed) -> Path:
    return output_dir(cfg) / "models" / ds / split / model / f"seed_{seed}"


def pred_path(cfg, exp, ds, split, model, seed) -> Path:
    return output_dir(cfg) / "preds" / exp / ds / split / model / f"seed_{seed}"


def _row(exp, ds, split, model, seed, m, extra=None, details=None):
    r = {"exp": exp, "dataset": ds, "split": split, "model": model, "seed": seed}
    r.update({k: m.get(k) for k in SUMMARY_KEYS})
    r["per_class"] = m.get("per_class")
    if details is not None:
        lo, hi = bootstrap_f1_ci(details, n_boot=500, seed=seed)
        r["f1_ci_low"], r["f1_ci_high"] = lo, hi
    r.update(extra or {})
    return r


def evaluate(cfg, gt, eps, test_flows, n_alert_flows):
    return episode_metrics(gt, eps, cfg["metrics"], test_flows, n_alert_flows, return_details=True)


# ---------------------------------------------------------------- baselines
def run_flow_baselines(cfg, exp, ds_name, parts, flow_kind, posts: dict, seed, tag_split) -> list[dict]:
    """Train one flow classifier, then apply several post-processors (e.g. xgb -> B1,B2,B3,B7)."""
    set_seed(seed)
    (tr, gtr), (va, gva), (te, gte) = parts
    classes = class_list(tr["attack"])
    with timer(f"[{ds_name}/{tag_split}] {flow_kind} seed {seed} fit"):
        params = tune_flow_model(flow_kind, classes, tr, va, cfg, seed)
        model = make_flow_model(flow_kind, classes, cfg, seed, params).fit(tr, va)
    mdir = model_dir(cfg, ds_name, tag_split, flow_kind, seed)
    model.save(mdir / ("model.pkl" if flow_kind == "xgb" else "model.pt"))
    Pv, Pt = model.predict_proba(va), model.predict_proba(te)
    flow_f1 = flow_macro_f1(encode(te["attack"], classes), Pt.argmax(1))
    n_alerts = int((Pt.argmax(1) != 0).sum())
    jacc = cfg["episodes"]["merge_victim_jaccard"]
    rows, post_params = [], {}
    for bname, post in posts.items():
        best, vf1 = tune_post(post, va, Pv, gva, classes, cfg)
        eps, _ = apply_post(post, te, Pt, classes, best, jacc)
        m, det = evaluate(cfg, gte, eps, te, n_alerts)
        save_df(eps, pred_path(cfg, exp, ds_name, tag_split, bname, seed))
        post_params[bname] = {"post": post, "params": best}
        rows.append(_row(exp, ds_name, tag_split, bname, seed, m,
                         {"flow_model": flow_kind, "post": post, "post_params": best, "val_f1": vf1,
                          "flow_f1_macro": flow_f1}, det))
        LOG.info("[%s/%s] %s seed %d: episode-F1 %.4f attr %.4f (flow-F1 %.4f)", ds_name, tag_split, bname, seed,
                 m["f1_macro"], m["attr_acc_macro"], flow_f1)
    save_json({"classes": classes, "posts": post_params}, mdir / "post.json")
    return rows


# ---------------------------------------------------------------- FlowDiar
def run_flowdiar(cfg, exp, ds_name, parts, seed, tag_split, variant="full", overrides=None,
                 save_model=True) -> dict:
    from .models.flowdiar import FlowDiar

    set_seed(seed)
    (tr, gtr), (va, gva), (te, gte) = parts
    classes = class_list(tr["attack"])
    b = cfg["flowdiar"]["bin_sec"]
    with timer(f"[{ds_name}/{tag_split}] host-bins"):
        hb_tr = build_hostbins(tr, b, gtr, classes)
        hb_va = build_hostbins(va, b, gva, classes)
        hb_te = build_hostbins(te, b)
    name = "FlowDiar" if variant == "full" else f"FlowDiar-{variant}"
    with timer(f"[{ds_name}/{tag_split}] {name} seed {seed}"):
        model = FlowDiar(cfg, classes, seed, overrides).fit(hb_tr, hb_va, va, gva)
    if save_model:
        model.save(model_dir(cfg, ds_name, tag_split, name, seed) / "model.pt")
    eps, flow_lab = model.predict(hb_te, te)
    n_alerts = int((flow_lab != BENIGN).sum())
    m, det = evaluate(cfg, gte, eps, te, n_alerts)
    save_df(eps, pred_path(cfg, exp, ds_name, tag_split, name, seed))
    flow_f1 = flow_macro_f1_names(te["attack"].to_numpy(), flow_lab)
    LOG.info("[%s/%s] %s seed %d: episode-F1 %.4f attr %.4f", ds_name, tag_split, name, seed,
             m["f1_macro"], m["attr_acc_macro"])
    return _row(exp, ds_name, tag_split, name, seed, m,
                {"variant": variant, "decode_params": model.decode_params, "flow_f1_macro": flow_f1}, det)


# ---------------------------------------------------------------- cross-dataset view
def family_parts(cfg, parts, common_cols=None):
    """Map attack names to families (cross-dataset) and keep shared feature columns."""
    fmap = family_map(cfg["registry"]["families"])
    out = []
    for f, e in parts:
        f = f.copy()
        e = e.copy()
        f["attack"] = to_family(f["attack"], fmap)
        e["attack"] = to_family(e["attack"], fmap)
        if common_cols is not None:
            drop = [c for c in feature_columns(f) if c not in common_cols]
            f = f.drop(columns=drop)
        out.append((f, e))
    return out
