"""Pilot study: is the episode-level direction worth pursuing?

Short version of the H1 gate (E1-E3) with one seed and no hyper-parameter search, plus
one extra measurement that decides how strong the research story is:

  ORACLE = true flow labels + the B1 grouping rule (tau tuned on val).
  * oracle close to 1.0  -> grouping is easy once flows are labelled correctly; any gap of
                            B1 comes from per-flow mistakes (weaker story for an episode model)
  * oracle clearly < 1.0 -> even perfect flow labels + time-gap grouping gets boundaries or
                            attribution wrong: the episode problem exists on its own

Checks are reported as PASS / WARN / FAIL with a final GO / STOP / UNCLEAR.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .config import output_dir
from .data.episodes import episodes_from_flow_labels
from .data.prepare import Dataset
from .metrics.episode import episode_metrics
from .metrics.errors import categorize, summarize
from .pipeline import run_flow_baselines, run_flowdiar
from .utils import LOG, save_json, to_set

PARTS = ("train", "val", "test")


def _oracle(cfg, parts):
    (_, _), (va, gva), (te, gte) = parts
    jacc = cfg["episodes"]["merge_victim_jaccard"]
    best_tau, best = None, -1.0
    for tau in cfg["post"]["sessionize"]["tau_grid_sec"]:
        eps, _ = episodes_from_flow_labels(va, va["attack"].to_numpy(), tau * 1000, jacc)
        f1 = episode_metrics(gva, eps, cfg["metrics"])["f1_macro"]
        if f1 > best:
            best_tau, best = tau, f1
    eps, _ = episodes_from_flow_labels(te, te["attack"].to_numpy(), best_tau * 1000, jacc)
    m = episode_metrics(gte, eps, cfg["metrics"], te)
    return m, best_tau, eps


def _overlap_stats(eps: pd.DataFrame):
    o = np.argsort(eps["t_start"].to_numpy())
    s, e = eps["t_start"].to_numpy()[o], eps["t_end"].to_numpy()[o]
    ov = np.zeros(len(eps), bool)
    for i in range(len(eps)):
        j = i + 1
        while j < len(eps) and s[j] <= e[i]:
            ov[i] = ov[j] = True
            j += 1
    return float(ov.mean()) if len(eps) else 0.0


def pilot_dataset(cfg: dict, name: str, seed: int = 0, with_flowdiar: bool = False, models=("B1", "B2", "B7")):
    cfg = copy.deepcopy(cfg)
    cfg["tuning"]["enabled"] = False
    ds = Dataset(cfg, name)
    checks = []

    def add(cid, title, status, value, note=""):
        checks.append({"check": cid, "noi_dung": title, "trang_thai": status, "gia_tri": value, "ghi_chu": note})

    # C1 ground truth independence
    indep = bool(ds.meta["gt_independent"])
    add("C1", "Ground truth độc lập với cách gom của B1", "PASS" if indep else "FAIL", ds.meta["gt_source"],
        "" if indep else "điền dataset/schedules/<tên>.csv rồi chạy lại prepare --force")

    # C2 sample size
    parts = [ds.part("temporal", p) for p in PARTS]
    n_eps = {p: len(parts[i][1]) for i, p in enumerate(PARTS)}
    per_cls_test = parts[2][1]["attack"].value_counts().to_dict()
    st = "PASS" if n_eps["test"] >= 20 else ("WARN" if n_eps["test"] >= 8 else "FAIL")
    add("C2", "Số đợt ở train/val/test", st, n_eps, f"test theo loại: {per_cls_test}")

    # C3 overlap
    ov = _overlap_stats(ds.episodes)
    add("C3", "Tỉ lệ đợt chồng thời gian với đợt khác", "PASS" if ov >= 0.05 else "WARN", round(ov, 4),
        "" if ov >= 0.05 else "gần như không chồng lấn -> bỏ giải mã nhiều track")

    # C4 attribution non-trivial
    att = set().union(*[to_set(a) for a in ds.episodes["attackers"]]) if len(ds.episodes) else set()
    add("C4", "Số IP tấn công khác nhau", "PASS" if len(att) >= 5 else "WARN", len(att),
        "" if len(att) >= 5 else "ít kẻ tấn công -> quy kết dễ, cần split tách host")

    # C5 oracle ceiling of the grouping rule
    om, otau, oeps = _oracle(cfg, parts)
    ocat = categorize(parts[2][1], oeps, cfg["metrics"])
    oshare = ocat["category"].value_counts(normalize=True).round(3).to_dict() if len(ocat) else {}
    add("C5", "Trần ORACLE: nhãn flow đúng + gom theo tau", "INFO",
        {"f1": round(om["f1_macro"], 4), "precision": round(om["precision_macro"], 4),
         "recall": round(om["recall_macro"], 4), "attr": round(om["attr_acc_macro"], 4), "tau_sec": otau,
         "loi_oracle": oshare},
        "oracle < 0.95: gom theo khoảng hở thời gian sai ngay cả khi nhãn flow đúng (xem loi_oracle: "
        "merged = đợt chồng/liền nhau bị gộp, fragmented = một đợt bị cắt vụn; precision thấp = "
        "đợt thật bị cắt thành nhiều mảnh thừa)" if om["f1_macro"] < 0.95
        else "oracle cao: khoảng trống (nếu có) đến từ lỗi phân loại flow")

    # C6 baselines on the temporal split
    xgb_posts = {b: cfg["baselines"][b]["post"] for b in models if cfg["baselines"][b]["flow_model"] == "xgb"}
    rows = run_flow_baselines(cfg, "PILOT", name, parts, "xgb", xgb_posts, seed, "temporal")
    res = {r["model"]: r for r in rows}
    best = max(res, key=lambda k: res[k]["f1_macro"])
    bf1, battr = res[best]["f1_macro"], res[best]["attr_acc_macro"]
    for b, r in res.items():
        add(f"C6-{b}", f"{b} split thời gian (F1 đợt / quy kết / F1 flow)", "INFO",
            {"f1": round(r["f1_macro"], 4), "precision": round(r["precision_macro"], 4),
             "recall": round(r["recall_macro"], 4), "attr": round(r["attr_acc_macro"], 4),
             "flow_f1": round(r["flow_f1_macro"], 4), "ttd_med_s": r["ttd_median_s"]})

    # C7 host-disjoint split
    host = None
    if "host" in ds.splits():
        hparts = [ds.part("host", p) for p in PARTS]
        if len(hparts[2][1]):
            hr = run_flow_baselines(cfg, "PILOT", name, hparts, "xgb", {"B1": "sessionize"}, seed, "host")[0]
            host = hr
            add("C7", "B1 split tách host tấn công", "INFO",
                {"f1": round(hr["f1_macro"], 4), "attr": round(hr["attr_acc_macro"], 4)})

    # C8 error analysis of the best baseline
    from .pipeline import pred_path
    from .utils import load_df

    cat = categorize(parts[2][1], load_df(pred_path(cfg, "PILOT", name, "temporal", best, seed)), cfg["metrics"])
    err = summarize(cat)
    share = cat["category"].value_counts(normalize=True).round(3).to_dict() if len(cat) else {}
    add("C8", f"Phân loại lỗi của {best}", "INFO", share)

    # C9 optional quick FlowDiar
    fd = None
    if with_flowdiar:
        try:
            fd = run_flowdiar(cfg, "PILOT", name, parts, seed, "temporal", save_model=False)
            add("C9", "FlowDiar nhanh (1 seed)", "INFO",
                {"f1": round(fd["f1_macro"], 4), "attr": round(fd["attr_acc_macro"], 4),
                 "diff_vs_best_points": round(100 * (fd["f1_macro"] - bf1), 2)})
        except ImportError as e:
            add("C9", "FlowDiar nhanh", "WARN", "bỏ qua", f"thiếu thư viện: {e}")

    # verdict
    g = cfg["gate"]
    gap_t = 100 * (1 - min(bf1, battr if battr == battr else 1.0))
    gap_h = 100 * (1 - min(host["f1_macro"], host["attr_acc_macro"] if host["attr_acc_macro"] == host["attr_acc_macro"]
                           else 1.0)) if host else 0.0
    if not indep:
        verdict = "KHÔNG KẾT LUẬN ĐƯỢC: ground truth là heuristic"
    elif n_eps["test"] < 8:
        verdict = "KHÔNG ĐỦ DỮ LIỆU: quá ít đợt ở test"
    elif bf1 >= g["stop_f1"] and (battr != battr or battr >= g["stop_attr"]) and (host is None or host["f1_macro"] >= g["stop_f1"]):
        verdict = "STOP: baseline đã gần hoàn hảo ở mức đợt"
    elif max(gap_t, gap_h) >= g["go_gap_points"]:
        src = "gom đợt" if om["f1_macro"] < 0.95 else "phân loại flow"
        verdict = f"GO: còn khoảng trống {max(gap_t, gap_h):.1f} điểm (nguồn chính: {src})"
    else:
        verdict = f"CHƯA RÕ: khoảng trống {max(gap_t, gap_h):.1f} điểm < {g['go_gap_points']}"
    out = {"dataset": name, "verdict": verdict, "best_baseline": best, "gap_temporal_points": gap_t,
           "gap_host_points": gap_h, "oracle_f1": om["f1_macro"], "checks": checks,
           "errors_by_class": err.to_dict(orient="records") if len(err) else []}
    save_json(out, output_dir(cfg) / "feasibility" / f"{name}.json")
    LOG.info("[%s] %s", name, verdict)
    return out


def write_summary(cfg: dict, results: list[dict]) -> str:
    lines = ["# Kết quả thử khả thi (pilot)", ""]
    lines += ["| Bộ dữ liệu | Kết luận | Baseline tốt nhất | Khoảng trống (thời gian / tách host) | Trần oracle F1 |",
              "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['dataset']} | {r['verdict']} | {r['best_baseline']} | "
                     f"{r['gap_temporal_points']:.1f} / {r['gap_host_points']:.1f} | {r['oracle_f1']:.3f} |")
    for r in results:
        lines += ["", f"## {r['dataset']}", "", "| Mã | Nội dung | Trạng thái | Giá trị | Ghi chú |", "|---|---|---|---|---|"]
        for c in r["checks"]:
            lines.append(f"| {c['check']} | {c['noi_dung']} | {c['trang_thai']} | {c['gia_tri']} | {c['ghi_chu']} |")
    lines += ["", "Cách đọc: GO khi baseline mạnh còn kém >= 10 điểm F1 đợt hoặc quy kết ở split thời gian hay tách host,",
              "STOP khi baseline >= 95% ở mọi split. Trần oracle cho biết khoảng trống đến từ gom đợt hay từ phân loại flow."]
    text = "\n".join(lines)
    p = output_dir(cfg) / "feasibility" / "summary.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return text
