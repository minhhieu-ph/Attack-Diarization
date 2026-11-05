"""Synthetic NetFlow-v3-like and CIDDS-like data for smoke tests.

The generator plants attack episodes (DoS, DDoS, scanning, brute force) with known
attackers, victims and times, some overlapping, over a benign background. It writes
files in the exact raw formats the real loaders expect, plus a schedule CSV.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

T0 = 1_700_000_000_000  # ms


def _benign(rng, n_hosts, n_servers, hours, rate_per_host_min):
    hosts = [f"10.0.0.{i}" for i in range(1, n_hosts + 1)]
    servers = [f"10.0.1.{i}" for i in range(1, n_servers + 1)]
    span = int(hours * 3600 * 1000)
    rows = []
    for h in hosts:
        n = rng.poisson(rate_per_host_min * hours * 60)
        ts = np.sort(rng.integers(0, span, n))
        for t in ts:
            dur = int(rng.exponential(800))
            rows.append((T0 + t, T0 + t + dur, h, rng.choice(servers), int(rng.integers(32768, 60999)),
                         int(rng.choice([80, 443, 53, 22, 25])), 6 if rng.random() < .8 else 17,
                         int(rng.lognormal(7, 1)), int(rng.lognormal(8, 1.2)),
                         int(rng.integers(2, 30)), int(rng.integers(2, 40)), dur, "Benign"))
    return rows, hosts, servers, span


def _episode(rng, kind, start, length, attackers, victims):
    rows = []
    if kind == "DoS":
        n = int(length / 300)
        for t in np.sort(rng.integers(start, start + length, n)):
            rows.append((t, t + 5, attackers[0], victims[0], int(rng.integers(1024, 65000)), 80, 6,
                         int(rng.integers(40, 120)), 0, 1, 0, 5, kind))
    elif kind == "DDoS":
        n = int(length / 150)
        for t in np.sort(rng.integers(start, start + length, n)):
            rows.append((t, t + 3, rng.choice(attackers), victims[0], int(rng.integers(1024, 65000)), 80, 17,
                         int(rng.integers(40, 80)), 0, 1, 0, 3, kind))
    elif kind == "scanning":
        n = int(length / 100)
        for t in np.sort(rng.integers(start, start + length, n)):
            rows.append((t, t + 1, attackers[0], rng.choice(victims), int(rng.integers(1024, 65000)),
                         int(rng.integers(1, 1024)), 6, 44, 0, 1, 0, 1, kind))
    elif kind == "bruteforce":
        n = int(length / 900)
        for t in np.sort(rng.integers(start, start + length, n)):
            d = int(rng.integers(200, 900))
            rows.append((t, t + d, attackers[0], victims[0], int(rng.integers(1024, 65000)), 22, 6,
                         int(rng.integers(1500, 3000)), int(rng.integers(2000, 4000)), 12, 10, d, kind))
    return rows


def _plan(rng, hosts, servers, span, n_episodes):
    kinds = ["DoS", "DDoS", "scanning", "bruteforce"]
    ext = [f"172.16.{i}.{j}" for i in range(1, 4) for j in range(1, 6)]  # attacker pool
    plan = []
    t = int(0.02 * span)
    for k in range(n_episodes):
        kind = kinds[k % len(kinds)]
        length = int(rng.integers(60_000, 300_000))
        if kind == "DDoS":
            att = list(rng.choice(ext, 4, replace=False))
            vic = [str(rng.choice(servers))]
        elif kind == "scanning":
            att = [str(rng.choice(ext))]
            vic = list(rng.choice(hosts + servers, 20, replace=False))
        else:
            att = [str(rng.choice(ext))]
            vic = [str(rng.choice(servers))]
        plan.append((kind, t, length, att, vic))
        # ~25% of episodes overlap with the next one
        t += length + (int(rng.integers(-length // 2, 0)) if rng.random() < 0.25 else int(rng.integers(60_000, 400_000)))
        if t > span * 0.97:
            break
    return plan


def make_synthetic_netflow(out_raw: Path, sched_path: Path, seed: int = 0, hours: float = 3.0,
                           n_episodes: int = 40) -> None:
    rng = np.random.default_rng(seed)
    rows, hosts, servers, span = _benign(rng, 40, 8, hours, 3.0)
    plan = _plan(rng, hosts, servers, span, n_episodes)
    sched = []
    for kind, s, L, att, vic in plan:
        ep = _episode(rng, kind, s, L, att, vic)
        rows += [(T0 + r[0], T0 + r[1], *r[2:]) for r in ep]
        sched.append({"start": T0 + s, "end": T0 + s + L, "attack": kind,
                      "attackers": ";".join(att), "victims": ";".join(vic) if kind != "scanning" else ""})
    cols = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT",
            "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS", "Attack"]
    df = pd.DataFrame(rows, columns=cols).sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    df["Label"] = (df["Attack"] != "Benign").astype(int)
    df["L7_PROTO"] = (df["L4_DST_PORT"] % 7).astype(float)
    df["TCP_FLAGS"] = np.where(df["PROTOCOL"] == 6, rng.integers(2, 31, len(df)), 0)
    df["SRC_TO_DST_IAT_AVG"] = (df["FLOW_DURATION_MILLISECONDS"] / df["IN_PKTS"].clip(lower=1)).round(2)
    out_raw.mkdir(parents=True, exist_ok=True)
    csv = df.to_csv(index=False)
    with zipfile.ZipFile(out_raw / "synth_netflow.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("NF-SYNTH-v3.csv", csv)
        z.writestr("NetFlow_v3_Features.csv", "Feature,Description\nIN_BYTES,bytes\n")  # decoy file
    sched_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sched).to_csv(sched_path, index=False)


def make_synthetic_cidds(out_raw: Path, seed: int = 1, hours: float = 3.0, n_episodes: int = 40) -> None:
    rng = np.random.default_rng(seed)
    rows, hosts, servers, span = _benign(rng, 30, 6, hours, 3.0)
    plan = _plan(rng, hosts, servers, span, n_episodes)
    rec = []
    for r in rows:
        rec.append((r[0] - T0, r[1] - T0, r[2], r[3], r[4], r[5], r[6], r[7], r[9], "normal", "---", "---"))
    names = {"DoS": "dos", "DDoS": "dos", "scanning": "portScan", "bruteforce": "bruteForce"}
    for k, (kind, s, L, att, vic) in enumerate(plan):
        for r in _episode(rng, kind, s, L, att, vic):
            rec.append((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[9], "attacker", names[kind], str(k + 1)))
            # reply flow victim -> attacker (dropped by the loader by default)
            rec.append((r[0] + 1, r[1] + 1, r[3], r[2], r[5], r[4], r[6], 40, 1, "victim", names[kind], str(k + 1)))
    d = pd.DataFrame(rec, columns=["ts", "te", "src", "dst", "sp", "dp", "proto", "bytes", "pkts", "class",
                                   "attackType", "attackID"]).sort_values("ts")
    t = pd.to_datetime(T0 + d["ts"], unit="ms")
    big = d["bytes"] > 1_000_000
    out = pd.DataFrame({
        "Date first seen": t.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
        "Duration": ((d["te"] - d["ts"]) / 1000).round(3),
        "Proto": np.where(d["proto"] == 6, "TCP  ", "UDP  "),
        "Src IP Addr": d["src"], "Src Pt": d["sp"], "Dst IP Addr": d["dst"], "Dst Pt": d["dp"].astype(float),
        "Packets": d["pkts"],
        "Bytes": np.where(big, (d["bytes"] / 1e6).round(1).astype(str) + " M", d["bytes"].astype(str)),
        "Flows": 1, "Flags": np.where(d["proto"] == 6, ".AP.S.", "......"), "Tos": 0,
        "class": d["class"], "attackType": d["attackType"], "attackID": d["attackID"], "attackDescription": "---",
    })
    p = out_raw / "traffic" / "OpenStack"
    p.mkdir(parents=True, exist_ok=True)
    out.to_csv(p / "CIDDS-001-internal-week1.csv", index=False)
