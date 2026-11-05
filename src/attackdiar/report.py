"""Aggregate results/*.jsonl into tables, the RQ1 gate decision and H2-H5 checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import output_dir
from .metrics.stats import paired_permutation_test
from .utils import LOG, has_module, read_jsonl


def _md(df: pd.DataFrame) -> str:
    if has_module("tabulate"):
        return df.to_markdown(index=False)
    return "```\n" + df.to_string(index=False) + "\n```"

COLS = ["f1_macro", "attr_acc_macro", "eer", "detect_rate", "ttd_median_s", "fa_per_hour", "flow_f1_macro"]


def load(cfg, exp) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(output_dir(cfg) / "results" / f"{exp}.jsonl"))


def mean_std(df: pd.DataFrame, keys=("dataset", "split", "model"), cols=COLS) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(list(keys))[cols]
    m, s = g.mean(), g.std()
    out = m.copy()
    for c in cols:
        out[c] = [f"{a:.4f} ± {b:.4f}" if pd.notna(b) else (f"{a:.4f}" if pd.notna(a) else "") for a, b in zip(m[c], s[c])]
    out["n_seeds"] = g.size()
    return out.reset_index()


def gate(cfg) -> dict:
    """H1 / RQ1: STOP if the best baseline is near-perfect on every dataset."""
    e2 = load(cfg, "E2")
    if e2.empty:
        return {"decision": "NO_DATA"}
    g = cfg["gate"]
    e1 = {r["dataset"]: r for r in read_jsonl_dir(cfg, "E1")}
    per = []
    for ds, r in e2.groupby("dataset"):
        mean = r.groupby("model")[["f1_macro", "attr_acc_macro"]].mean()
        best = mean["f1_macro"].idxmax()
        f1, attr = mean.loc[best, "f1_macro"], mean.loc[best, "attr_acc_macro"]
        per.append({"dataset": ds, "best_baseline": best, "f1": f1, "attr": attr,
                    "near_perfect": bool(f1 >= g["stop_f1"] and (np.isnan(attr) or attr >= g["stop_attr"])),
                    "gap_points": 100 * (1 - min(f1, attr if not np.isnan(attr) else 1.0)),
                    "gt_independent": e1.get(ds, {}).get("gt_independent")})
    per = pd.DataFrame(per)
    if (per["gt_independent"] == False).any():  # noqa: E712
        decision = "INVALID: heuristic ground truth on " + ", ".join(per.loc[per["gt_independent"] == False, "dataset"])  # noqa: E712
    elif per["near_perfect"].all():
        decision = "STOP (H1 rejected: baselines >= thresholds on every dataset)"
    elif (per["gap_points"] >= g["go_gap_points"]).any():
        decision = "GO (gap >= %d points on: %s)" % (g["go_gap_points"], ", ".join(
            per.loc[per["gap_points"] >= g["go_gap_points"], "dataset"]))
    else:
        decision = "UNCLEAR (gap below %d points; check host split and error analysis)" % g["go_gap_points"]
    return {"decision": decision, "table": per}


def read_jsonl_dir(cfg, exp):
    import json

    d = output_dir(cfg) / exp
    return [json.load(open(p)) for p in sorted(d.glob("*.json"))] if d.exists() else []


def h2(cfg) -> pd.DataFrame:
    e2, e4, e8 = load(cfg, "E2"), load(cfg, "E4"), load(cfg, "E8")
    if e2.empty or e4.empty:
        return pd.DataFrame()
    h = cfg["hypotheses"]
    rows = []
    for ds, r in e4.groupby("dataset"):
        b = e2[e2["dataset"] == ds]
        if b.empty:
            continue
        best = b.groupby("model")["f1_macro"].mean().idxmax()
        bb = b[b["model"] == best].set_index("seed")["f1_macro"]
        fd = r.set_index("seed")["f1_macro"]
        seeds = sorted(set(bb.index) & set(fd.index))
        diff = 100 * (fd[seeds].mean() - bb[seeds].mean())
        p = paired_permutation_test(fd[seeds].values, bb[seeds].values)
        row = {"dataset": ds, "best_baseline": best, "flowdiar_f1": fd.mean(), "baseline_f1": bb.mean(),
               "diff_points": diff, "p_value": p}
        if not e8.empty:
            t = e8[e8["dataset"] == ds].groupby("model")["ttd_median_s"].mean()
            if {"FlowDiar", "B1"} <= set(t.index) and t["B1"] > 0:
                row["ttd_reduction"] = 1 - t["FlowDiar"] / t["B1"]
        row["H2_supported"] = bool((diff >= h["h2_f1_points"] and p < 0.05)
                                   or row.get("ttd_reduction", 0) >= h["h2_ttd_reduction"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(cfg) -> str:
    out = output_dir(cfg)
    lines = ["# Results", ""]
    g = gate(cfg)
    lines += ["## Gate (H1)", "", f"**Decision:** {g['decision']}", ""]
    if "table" in g:
        lines += [_md(g["table"]), ""]
    for exp in ["E2", "E4", "E5", "E6"]:
        df = load(cfg, exp)
        if df.empty:
            continue
        t = mean_std(df)
        t.to_csv(out / "results" / f"{exp}_table.csv", index=False)
        lines += [f"## {exp}", "", _md(t), ""]
    hh = h2(cfg)
    if not hh.empty:
        hh.to_csv(out / "results" / "H2.csv", index=False)
        lines += ["## H2: FlowDiar vs best baseline (temporal split)", "", _md(hh), ""]
    for exp, keys, cols in [("E7", ("dataset", "split", "model"), ["recall_macro", "f1_macro", "attr_acc_macro"]),
                            ("E8", ("dataset", "model"), ["ttd_median_s", "ttd_p90_s", "detect_rate", "f1_macro"])]:
        df = load(cfg, exp)
        if not df.empty:
            t = mean_std(df, keys, cols)
            t.to_csv(out / "results" / f"{exp}_table.csv", index=False)
            lines += [f"## {exp}", "", _md(t), ""]
    p = out / "results" / "E3_errors.csv"
    if p.exists():
        lines += ["## E3: error categories of the best baseline", "", _md(pd.read_csv(p)), ""]
    text = "\n".join(lines)
    (out / "report.md").write_text(text)
    LOG.info("report -> %s\nGATE: %s", out / "report.md", g["decision"])
    return text
