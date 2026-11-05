"""Experiments E1-E8 (see the proposal). Each appends rows to outputs/results/<EXP>.jsonl
and skips (dataset, split, model, seed) combinations that are already there."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import output_dir
from .data.episodes import episodes_from_flow_labels
from .data.features import feature_columns
from .data.prepare import Dataset
from .data.stress import PERTURB
from .metrics.episode import episode_metrics
from .metrics.errors import categorize, summarize
from .pipeline import (model_dir, pred_path, family_parts, run_flow_baselines, run_flowdiar)
from .utils import LOG, append_jsonl, df_exists, load_df, load_json, read_jsonl, save_json, to_set

PARTS = ("train", "val", "test")


def _results(cfg, exp):
    return output_dir(cfg) / "results" / f"{exp}.jsonl"


def _done(cfg, exp) -> set:
    return {(r["dataset"], r["split"], r["model"], r["seed"]) for r in read_jsonl(_results(cfg, exp))}


def _parts(ds: Dataset, split: str):
    return [ds.part(split, p) for p in PARTS]


def _models(cfg) -> list[str]:
    return cfg.get("run", {}).get("models") or list(cfg["baselines"])


# ---------------------------------------------------------------- E1
def e1_ground_truth_stats(cfg):
    rows = []
    for name in cfg["data"]["datasets"]:
        ds = Dataset(cfg, name)
        eps, flows = ds.episodes, ds.flows
        dur = (eps["t_end"] - eps["t_start"]) / 1000.0
        n_att = eps["attackers"].map(lambda s: len(to_set(s)))
        n_vic = eps["victims"].map(lambda s: len(to_set(s)))
        o = np.argsort(eps["t_start"].to_numpy())
        s, e = eps["t_start"].to_numpy()[o], eps["t_end"].to_numpy()[o]
        hosts = [to_set(a) | to_set(v) for a, v in zip(eps["attackers"].to_numpy()[o], eps["victims"].to_numpy()[o])]
        overlap = np.zeros(len(eps), bool)
        overlap_host = np.zeros(len(eps), bool)
        for i in range(len(eps)):
            j = i + 1
            while j < len(eps) and s[j] <= e[i]:
                overlap[i] = overlap[j] = True
                if hosts[i] & hosts[j]:
                    overlap_host[i] = overlap_host[j] = True
                j += 1
        row = {"dataset": name, "gt_source": ds.meta["gt_source"], "gt_independent": ds.meta["gt_independent"],
               "n_flows": len(flows), "n_attack_flows": int(flows["label"].sum()),
               "unassigned_attack_flows": int(((flows["label"] == 1) & (flows["episode_id"] < 0)).sum()),
               "n_episodes": len(eps), "episodes_per_class": eps["attack"].value_counts().to_dict(),
               "duration_s_median": float(dur.median()) if len(eps) else None,
               "duration_s_p90": float(dur.quantile(0.9)) if len(eps) else None,
               "attackers_median": float(n_att.median()) if len(eps) else None,
               "victims_median": float(n_vic.median()) if len(eps) else None,
               "distinct_attackers": len(set().union(*[to_set(a) for a in eps["attackers"]])) if len(eps) else 0,
               "overlap_ratio": float(overlap.mean()) if len(eps) else 0.0,
               "overlap_shared_host_ratio": float(overlap_host.mean()) if len(eps) else 0.0}
        # cross-check: heuristic episodes from the TRUE flow labels vs GT
        for tau in cfg["episodes"]["tau_grid_sec"]:
            heps, _ = episodes_from_flow_labels(flows, flows["attack"].to_numpy(), tau * 1000,
                                                cfg["episodes"]["merge_victim_jaccard"])
            m = episode_metrics(eps, heps, cfg["metrics"])
            row[f"heuristic_tau{tau}_f1"] = m["f1_macro"]
            row[f"heuristic_tau{tau}_attr"] = m["attr_acc_macro"]
        LOG.info("E1 %s: %s", name, {k: v for k, v in row.items() if k != "episodes_per_class"})
        rows.append(row)
        save_json(row, output_dir(cfg) / "E1" / f"{name}.json")
    pd.DataFrame(rows).to_csv(output_dir(cfg) / "E1" / "summary.csv", index=False)
    return rows


# ---------------------------------------------------------------- E2
def e2_baselines(cfg):
    """All baselines on the temporal split (E5 runs them on the host split)."""
    e2_baselines_into(cfg, "E2", "temporal")


# ---------------------------------------------------------------- E3
def e3_error_analysis(cfg, split="temporal"):
    res = pd.DataFrame(read_jsonl(_results(cfg, "E2")))
    if res.empty:
        LOG.error("E3 needs E2 results")
        return
    out = []
    for name in cfg["data"]["datasets"]:
        r = res[(res["dataset"] == name) & (res["split"] == split)]
        if r.empty:
            continue
        best = r.groupby("model")["f1_macro"].mean().idxmax()
        ds = Dataset(cfg, name)
        _, gte = ds.part(split, "test")
        cats = []
        for seed in sorted(r["seed"].unique()):
            p = pred_path(cfg, "E2", name, split, best, seed)
            if df_exists(p):
                c = categorize(gte, load_df(p), cfg["metrics"])
                cats.append(c.assign(seed=seed))
        if not cats:
            continue
        cat = pd.concat(cats, ignore_index=True)
        s = summarize(cat).assign(dataset=name, model=best)
        out.append(s)
        LOG.info("E3 %s (best baseline %s):\n%s", name, best, s.to_string(index=False))
    if out:
        pd.concat(out).to_csv(output_dir(cfg) / "results" / "E3_errors.csv", index=False)


# ---------------------------------------------------------------- E4 / E6
def e4_flowdiar(cfg, split="temporal", exp="E4", variant="full", overrides=None):
    done = _done(cfg, exp)
    model = "FlowDiar" if variant == "full" else f"FlowDiar-{variant}"
    for name in cfg["data"]["datasets"]:
        ds = Dataset(cfg, name)
        if split not in ds.splits():
            continue
        parts = None
        for seed in cfg["seeds"]:
            if (name, split, model, seed) in done:
                continue
            parts = parts or _parts(ds, split)
            append_jsonl(run_flowdiar(cfg, exp, name, parts, seed, split, variant, overrides), _results(cfg, exp))


ABLATIONS = {"no_graph": {"use_graph": False}, "no_temporal": {"use_temporal": False},
             "sessionize_decoder": {"decoder": "sessionize"}}


def e6_ablation(cfg):
    for v, ov in ABLATIONS.items():
        e4_flowdiar(cfg, "temporal", "E6", v, ov)


# ---------------------------------------------------------------- E5
def e5_generalization(cfg):
    # (a) host-disjoint split: FlowDiar + all baselines
    e4_flowdiar(cfg, "host", "E5")
    e2_baselines_into(cfg, "E5", "host")
    # (b) cross-dataset among datasets with the same loader (shared features)
    done = _done(cfg, "E5")
    names = cfg["data"]["datasets"]
    loaders = {n: cfg["registry"][n]["loader"] for n in names}
    cache = {}
    for a in names:
        for b in names:
            if a == b or loaders[a] != loaders[b]:
                continue
            tag = f"cross:{a}->{b}"
            da = cache.setdefault(a, Dataset(cfg, a))
            db = cache.setdefault(b, Dataset(cfg, b))
            common = sorted(set(feature_columns(da.flows)) & set(feature_columns(db.flows)))
            pa = [da.part("temporal", p) for p in ("train", "val")]
            pb = [db.part("temporal", "test")]
            parts = family_parts(cfg, pa + pb, common)
            for seed in cfg["seeds"]:
                if (b, tag, "FlowDiar", seed) not in done:
                    append_jsonl(run_flowdiar(cfg, "E5", b, parts, seed, tag, save_model=False), _results(cfg, "E5"))
                if (b, tag, "B1", seed) not in done:
                    for r in run_flow_baselines(cfg, "E5", b, parts, "xgb", {"B1": "sessionize"}, seed, tag):
                        append_jsonl(r, _results(cfg, "E5"))


def e2_baselines_into(cfg, exp, split):
    done = _done(cfg, exp)
    groups: dict[str, dict] = {}
    for bname in _models(cfg):
        if bname not in cfg["baselines"]:
            continue
        spec = cfg["baselines"][bname]
        groups.setdefault(spec["flow_model"], {})[bname] = spec["post"]
    for name in cfg["data"]["datasets"]:
        ds = Dataset(cfg, name)
        if split not in ds.splits():
            continue
        parts = None
        for seed in cfg["seeds"]:
            for kind, posts in groups.items():
                todo = {b: p for b, p in posts.items() if (name, split, b, seed) not in done}
                if todo:
                    parts = parts or _parts(ds, split)
                    for r in run_flow_baselines(cfg, exp, name, parts, kind, todo, seed, split):
                        append_jsonl(r, _results(cfg, exp))


# ---------------------------------------------------------------- E7 / E8
def _load_or_train(cfg, name, parts, seed):
    """FlowDiar and the B1 model trained on the temporal split (from E4/E2 or freshly)."""
    from .baselines.flow_models import XGBFlowModel
    from .models.flowdiar import FlowDiar

    fd_path = model_dir(cfg, name, "temporal", "FlowDiar", seed) / "model.pt"
    if not fd_path.exists():
        run_flowdiar(cfg, "E4", name, parts, seed, "temporal")
    fd = FlowDiar.load(fd_path, cfg)
    xdir = model_dir(cfg, name, "temporal", "xgb", seed)
    if not (xdir / "model.pkl").exists():
        run_flow_baselines(cfg, "E2", name, parts, "xgb", {"B1": "sessionize"}, seed, "temporal")
    xgb = XGBFlowModel.load(xdir / "model.pkl", cfg)
    post = load_json(xdir / "post.json")["posts"]["B1"]
    return fd, xgb, post


def _eval_both(cfg, fd, xgb, post, te, gte, mode="offline"):
    from .baselines.postprocess import apply_post
    from .data.hostbins import build_hostbins

    hb = build_hostbins(te, cfg["flowdiar"]["bin_sec"])
    eps_fd, lab = fd.predict(hb, te, mode)
    P = xgb.predict_proba(te)
    eps_b1, _ = apply_post(post["post"], te, P, xgb.classes, post["params"], cfg["episodes"]["merge_victim_jaccard"])
    m_fd = episode_metrics(gte, eps_fd, cfg["metrics"], te, int((lab != "Benign").sum()))
    m_b1 = episode_metrics(gte, eps_b1, cfg["metrics"], te, int((P.argmax(1) != 0).sum()))
    return m_fd, m_b1


def e7_stress(cfg):
    done = _done(cfg, "E7")
    sc = cfg["stress"]
    settings = ([("slow", f) for f in sc["slow_factors"]] + [("interleave", r) for r in sc["interleave_rates"]]
                + [("split_sources", k) for k in sc["split_sources"]])
    for name in cfg["data"]["datasets"]:
        ds = Dataset(cfg, name)
        parts = _parts(ds, "temporal")
        te, gte = parts[2]
        for seed in cfg["seeds"]:
            fd = None
            for kind, val in [("none", 0)] + settings:
                tag = f"stress:{kind}={val}"
                if (name, tag, "FlowDiar", seed) in done:
                    continue
                if fd is None:
                    fd, xgb, post = _load_or_train(cfg, name, parts, seed)
                f2, g2 = (te, gte) if kind == "none" else PERTURB[kind](te, gte, val, seed)
                m_fd, m_b1 = _eval_both(cfg, fd, xgb, post, f2, g2)
                for model, m in (("FlowDiar", m_fd), ("B1", m_b1)):
                    append_jsonl({"exp": "E7", "dataset": name, "split": tag, "model": model, "seed": seed,
                                  "perturbation": kind, "level": val,
                                  **{k: m[k] for k in ("f1_macro", "recall_macro", "attr_acc_macro", "detect_rate")}},
                                 _results(cfg, "E7"))
                LOG.info("E7 %s %s seed %d: FlowDiar recall %.3f | B1 recall %.3f", name, tag, seed,
                         m_fd["recall_macro"], m_b1["recall_macro"])


def e8_online(cfg):
    done = _done(cfg, "E8")
    for name in cfg["data"]["datasets"]:
        ds = Dataset(cfg, name)
        parts = _parts(ds, "temporal")
        te, gte = parts[2]
        for seed in cfg["seeds"]:
            if (name, "online", "FlowDiar", seed) in done:
                continue
            fd, xgb, post = _load_or_train(cfg, name, parts, seed)
            m_fd, m_b1 = _eval_both(cfg, fd, xgb, post, te, gte, mode="online")
            for model, m in (("FlowDiar", m_fd), ("B1", m_b1)):
                append_jsonl({"exp": "E8", "dataset": name, "split": "online", "model": model, "seed": seed,
                              **{k: m[k] for k in ("f1_macro", "detect_rate", "ttd_median_s", "ttd_p90_s",
                                                   "fa_per_hour")}}, _results(cfg, "E8"))
            LOG.info("E8 %s seed %d: TTD median FlowDiar %.1fs | B1 %.1fs", name, seed,
                     m_fd["ttd_median_s"], m_b1["ttd_median_s"])


EXPERIMENTS = {"E1": e1_ground_truth_stats, "E2": e2_baselines, "E3": e3_error_analysis, "E4": e4_flowdiar,
               "E5": e5_generalization, "E6": e6_ablation, "E7": e7_stress, "E8": e8_online}
