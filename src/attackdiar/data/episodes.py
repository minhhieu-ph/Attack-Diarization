"""Attack episodes: ground-truth construction and the shared flow->episode grouping.

An episode is one row with
    episode_id, attack, t_start, t_end (ms), attackers, victims (";"-joined IPs),
    n_flows, t_alert (ms; earliest time a flow of the episode was observable).

The same `sessionize` + `merge_multi_attacker` code is used for
  * the heuristic ground truth (only as a fallback / cross-check), and
  * baseline B1 (applied to *predicted* attack flows),
so any difference between them comes from the flow labels, not from code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import LOG, SEP, jaccard, to_set
from .loaders import normalize_attack

EP_COLS = ["episode_id", "attack", "t_start", "t_end", "attackers", "victims", "n_flows", "t_alert"]


def empty_episodes() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="int64" if c in ("episode_id", "t_start", "t_end", "n_flows", "t_alert")
                                      else "object") for c in EP_COLS})


def _join_unique(seg: np.ndarray, vals: np.ndarray) -> pd.Series:
    """For each segment id, ';'-join the sorted unique values."""
    d = pd.DataFrame({"seg": seg, "v": vals}).drop_duplicates()
    return d.sort_values(["seg", "v"]).groupby("seg")["v"].agg(SEP.join)


# ---------------------------------------------------------------- sessionize
def sessionize(t_start, t_end, src, dst, cls, gap_ms: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Group alert flows by (src, class) and split when the gap to the running
    end of the group exceeds gap_ms. Returns (episodes, seg id per input row)."""
    t_start = np.asarray(t_start, dtype=np.int64)
    t_end = np.asarray(t_end, dtype=np.int64)
    src = np.asarray(src, dtype=object)
    dst = np.asarray(dst, dtype=object)
    cls = np.asarray(cls, dtype=object)
    n = len(t_start)
    if n == 0:
        return empty_episodes(), np.empty(0, dtype=np.int64)
    sc, _ = pd.factorize(src)
    cc, _ = pd.factorize(cls)
    order = np.lexsort((t_start, cc, sc))
    ts, te, s_o, c_o = t_start[order], t_end[order], sc[order], cc[order]
    newgrp = np.r_[True, (s_o[1:] != s_o[:-1]) | (c_o[1:] != c_o[:-1])]
    gid = np.cumsum(newgrp) - 1
    runmax = pd.Series(te).groupby(gid).cummax().to_numpy()
    prev = np.r_[np.iinfo(np.int64).min // 2, runmax[:-1]]
    newseg = newgrp | ((ts - prev) > gap_ms)
    seg_sorted = np.cumsum(newseg) - 1
    seg = np.empty(n, dtype=np.int64)
    seg[order] = seg_sorted
    return segments_to_episodes(t_start, t_end, src, dst, cls, seg), seg


def segments_to_episodes(t_start, t_end, src, dst, cls, seg) -> pd.DataFrame:
    """Aggregate rows sharing a segment id into episodes (episode_id = segment id).
    Rows of one segment are expected to share a class (the first one is used)."""
    src = np.asarray(src, dtype=object)
    dst = np.asarray(dst, dtype=object)
    if len(seg) == 0:
        return empty_episodes()
    d = pd.DataFrame({"seg": seg, "ts": t_start, "te": t_end, "cls": cls})
    g = d.groupby("seg")
    eps = pd.DataFrame({
        "episode_id": g.size().index.to_numpy(),
        "attack": g["cls"].first().to_numpy(),
        "t_start": g["ts"].min().to_numpy(),
        "t_end": g["te"].max().to_numpy(),
        "n_flows": g.size().to_numpy(),
        "t_alert": g["te"].min().to_numpy(),
    })
    eps["attackers"] = _join_unique(seg, src).reindex(eps["episode_id"]).to_numpy()
    eps["victims"] = _join_unique(seg, dst).reindex(eps["episode_id"]).to_numpy()
    return eps[EP_COLS]


def merge_multi_attacker(eps: pd.DataFrame, seg: np.ndarray, jaccard_thr: float = 0.5,
                         gap_ms: float = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge same-class episodes that overlap in time and share victims
    (Jaccard >= thr) into one multi-attacker episode (e.g. DDoS)."""
    if len(eps) == 0:
        return eps, seg
    eps = eps.sort_values(["attack", "t_start"], kind="mergesort").reset_index(drop=True)
    parent = {}
    clusters = {}  # cid -> dict
    for cls, sub in eps.groupby("attack", sort=False):
        active: list[int] = []
        for row in sub.itertuples(index=False):
            v = to_set(row.victims)
            active = [c for c in active if clusters[c]["t_end"] >= row.t_start - gap_ms]
            best, best_j = None, jaccard_thr
            for c in active:
                j = jaccard(clusters[c]["victims"], v)
                if j >= best_j:
                    best, best_j = c, j
            if best is None:
                cid = row.episode_id
                clusters[cid] = {"attack": cls, "t_start": row.t_start, "t_end": row.t_end,
                                 "attackers": set(to_set(row.attackers)), "victims": set(v),
                                 "n_flows": row.n_flows, "t_alert": row.t_alert}
                active.append(cid)
                parent[row.episode_id] = cid
            else:
                c = clusters[best]
                c["t_end"] = max(c["t_end"], row.t_end)
                c["t_alert"] = min(c["t_alert"], row.t_alert)
                c["attackers"] |= to_set(row.attackers)
                c["victims"] |= v
                c["n_flows"] += row.n_flows
                parent[row.episode_id] = best
    new_ids = {cid: i for i, cid in enumerate(sorted(clusters, key=lambda c: clusters[c]["t_start"]))}
    out = pd.DataFrame([{
        "episode_id": new_ids[cid], "attack": c["attack"], "t_start": c["t_start"], "t_end": c["t_end"],
        "attackers": SEP.join(sorted(c["attackers"])), "victims": SEP.join(sorted(c["victims"])),
        "n_flows": c["n_flows"], "t_alert": c["t_alert"]} for cid, c in clusters.items()])
    out = out.sort_values("episode_id").reset_index(drop=True)[EP_COLS]
    mapping = {old: new_ids[cid] for old, cid in parent.items()}
    new_seg = pd.Series(seg).map(mapping).to_numpy(dtype=np.int64) if len(seg) else seg
    return out, new_seg


def episodes_from_flow_labels(flows: pd.DataFrame, cls: np.ndarray, gap_ms: float,
                              jaccard_thr: float = 0.5, merge: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Turn per-flow class labels (str array, "Benign" = no alert) into episodes.
    Returns episodes and an episode id per flow (-1 for non-alert flows)."""
    cls = np.asarray(cls, dtype=object)
    mask = cls != "Benign"
    flow_ep = np.full(len(flows), -1, dtype=np.int64)
    if not mask.any():
        return empty_episodes(), flow_ep
    f = flows.loc[mask]
    eps, seg = sessionize(f["t_start"].to_numpy(), f["t_end"].to_numpy(), f["src_ip"].to_numpy(),
                          f["dst_ip"].to_numpy(), cls[mask], gap_ms)
    if merge:
        eps, seg = merge_multi_attacker(eps, seg, jaccard_thr)
    flow_ep[np.flatnonzero(mask)] = seg
    return eps, flow_ep


# ---------------------------------------------------------------- ground truth
def gt_from_attack_id(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    att = (df["label"].to_numpy() == 1) & (df["attack_id"].to_numpy() != "")
    flow_ep = np.full(len(df), -1, dtype=np.int64)
    if not att.any():
        return empty_episodes(), flow_ep
    a = df.loc[att, ["t_start", "t_end", "src_ip", "dst_ip", "attack", "attack_id", "role"]]
    g = a.groupby("attack_id")
    eps = pd.DataFrame({
        "attack_id": g.size().index,
        "attack": g["attack"].agg(lambda s: s.mode().iloc[0]).to_numpy(),
        "t_start": g["t_start"].min().to_numpy(),
        "t_end": g["t_end"].max().to_numpy(),
        "n_flows": g.size().to_numpy(),
        "t_alert": g["t_end"].min().to_numpy(),
    })
    is_att = a["role"].str.lower().eq("attacker")
    src_att = a.loc[is_att].groupby("attack_id")["src_ip"].agg(lambda s: SEP.join(sorted(set(s))))
    dst_att = a.loc[is_att].groupby("attack_id")["dst_ip"].agg(lambda s: SEP.join(sorted(set(s))))
    src_all = g["src_ip"].agg(lambda s: SEP.join(sorted(set(s))))
    dst_all = g["dst_ip"].agg(lambda s: SEP.join(sorted(set(s))))
    eps["attackers"] = src_att.reindex(eps["attack_id"]).fillna(src_all.reindex(eps["attack_id"])).to_numpy()
    eps["victims"] = dst_att.reindex(eps["attack_id"]).fillna(dst_all.reindex(eps["attack_id"])).to_numpy()
    eps = eps.sort_values("t_start", kind="mergesort").reset_index(drop=True)
    eps["episode_id"] = np.arange(len(eps), dtype=np.int64)
    id_map = dict(zip(eps["attack_id"], eps["episode_id"]))
    flow_ep[np.flatnonzero(att)] = a["attack_id"].map(id_map).to_numpy()
    return eps[EP_COLS], flow_ep


def _parse_time(col: pd.Series, tz: str) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().all():
        return num.astype("int64")
    t = pd.to_datetime(col.astype(str), format="mixed")
    if t.dt.tz is None:
        t = t.dt.tz_localize(tz)
    return ((t.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")


def load_schedule(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, comment="#")
    s.columns = [c.strip().lower() for c in s.columns]
    need = {"start", "end", "attack"}
    if not need <= set(s.columns):
        raise ValueError(f"{path}: schedule needs columns {sorted(need)} (+ attackers, victims, tz)")
    tz = s["tz"].iloc[0] if "tz" in s.columns and len(s) else "UTC"
    s["start_ms"] = _parse_time(s["start"], tz)
    s["end_ms"] = _parse_time(s["end"], tz)
    s["attack"] = normalize_attack(s["attack"])
    for c in ("attackers", "victims"):
        s[c] = s[c].fillna("").astype(str) if c in s.columns else ""
    return s


def gt_from_schedule(df: pd.DataFrame, sched: pd.DataFrame, tol_ms: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Each schedule row is one episode. Attack flows are assigned to a row when the
    class matches, the start time lies in [start - tol, end + tol] and the hosts match
    the listed attackers/victims (when given)."""
    flow_ep = np.full(len(df), -1, dtype=np.int64)
    is_att = df["label"].to_numpy() == 1
    att_lower = df["attack"].str.lower().to_numpy()
    ts = df["t_start"].to_numpy()
    src = df["src_ip"].to_numpy()
    dst = df["dst_ip"].to_numpy()
    rows, conflicts = [], 0
    for i, r in enumerate(sched.itertuples(index=False)):
        lo, hi = r.start_ms - tol_ms, r.end_ms + tol_ms
        i0, i1 = np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "right")
        idx = np.arange(i0, i1)
        m = is_att[idx] & (att_lower[idx] == str(r.attack).lower())
        A, V = to_set(r.attackers), to_set(r.victims)
        if A:
            m &= np.isin(src[idx], list(A))
        if V:
            m &= np.isin(dst[idx], list(V))
        idx = idx[m]
        conflicts += int((flow_ep[idx] >= 0).sum())
        idx = idx[flow_ep[idx] < 0]
        if len(idx) == 0:
            continue
        eid = len(rows)
        flow_ep[idx] = eid
        sub = df.iloc[idx]
        rows.append({"episode_id": eid, "attack": r.attack,
                     "t_start": int(sub["t_start"].min()), "t_end": int(sub["t_end"].max()),
                     "attackers": SEP.join(sorted(A & set(sub["src_ip"]))) if A else SEP.join(sorted(set(sub["src_ip"]))),
                     "victims": SEP.join(sorted(set(sub["dst_ip"]))),
                     "n_flows": len(idx), "t_alert": int(sub["t_end"].min())})
    eps = pd.DataFrame(rows, columns=EP_COLS) if rows else empty_episodes()
    info = {"schedule_rows": len(sched), "episodes": len(eps), "flows_in_two_rows": conflicts,
            "unassigned_attack_flows": int((is_att & (flow_ep < 0)).sum())}
    return eps, flow_ep, info


def build_ground_truth(df: pd.DataFrame, cfg: dict, name: str, ds_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    ecfg = cfg["episodes"]
    sched_path = ds_dir / "schedules" / f"{name}.csv"
    for source in ecfg["source_priority"]:
        if source == "attack_id" and (df["attack_id"] != "").any():
            eps, fe = gt_from_attack_id(df)
            meta = {"gt_source": "attack_id", "gt_independent": True}
            break
        if source == "schedule" and sched_path.exists():
            eps, fe, info = gt_from_schedule(df, load_schedule(sched_path), int(ecfg["schedule_tolerance_sec"] * 1000))
            meta = {"gt_source": "schedule", "gt_independent": True, **info}
            if info["unassigned_attack_flows"]:
                LOG.warning("[%s] %d attack flows not covered by the schedule (excluded from GT)",
                            name, info["unassigned_attack_flows"])
            break
        if source == "heuristic":
            LOG.warning("[%s] no attack_id and no %s -> HEURISTIC ground truth (tau=%ss). "
                        "This is NOT independent of baseline B1; fill the schedule before testing H1.",
                        name, sched_path, ecfg["tau_sec"])
            eps, fe = episodes_from_flow_labels(df, df["attack"].to_numpy(), ecfg["tau_sec"] * 1000,
                                                ecfg["merge_victim_jaccard"])
            meta = {"gt_source": "heuristic", "gt_independent": False, "tau_sec": ecfg["tau_sec"]}
            break
    else:
        raise RuntimeError(f"[{name}] no ground-truth source available")
    meta["n_episodes"] = len(eps)
    return eps, fe, meta


def restrict_episodes(eps: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """Episodes as seen inside a subset of flows (times clipped to observed flows)."""
    sub = flows.loc[flows["episode_id"] >= 0, ["episode_id", "t_start", "t_end"]]
    if sub.empty:
        return empty_episodes()
    g = sub.groupby("episode_id")
    agg = pd.DataFrame({"t_start": g["t_start"].min(), "t_end": g["t_end"].max(),
                        "n_flows": g.size(), "t_alert": g["t_end"].min()})
    base = eps.set_index("episode_id").loc[agg.index, ["attack", "attackers", "victims"]]
    out = pd.concat([base, agg], axis=1).reset_index()
    return out[EP_COLS].sort_values("t_start", kind="mergesort").reset_index(drop=True)
