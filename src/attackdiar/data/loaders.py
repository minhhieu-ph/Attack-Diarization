"""Parse raw files into one canonical flow table per dataset.

Canonical columns (one row = one flow record):
    t_start, t_end      int64 epoch milliseconds
    src_ip, dst_ip      str
    src_port, dst_port  int
    protocol            int (IANA number)
    bytes_fwd/bwd, pkts_fwd/bwd, duration_ms   float (used by FlowDiar host-bin features)
    label               int 0/1
    attack              str, "Benign" for benign
    attack_id           str, "" when the dataset has no native attack id
    role                str, CIDDS class (attacker/victim/normal/...) or ""
    f_*                 numeric per-flow features for flow classifiers
                        (no IPs, no absolute timestamps, no source port)
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import LOG, has_module

BENIGN_NAMES = {"benign", "normal", "---", "", "nan", "none"}
CANON = ["t_start", "t_end", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
         "bytes_fwd", "bytes_bwd", "pkts_fwd", "pkts_bwd", "duration_ms",
         "label", "attack", "attack_id", "role"]


def normalize_attack(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.where(~s.str.lower().isin(BENIGN_NAMES), "Benign")


# ---------------------------------------------------------------- file discovery
def _read_csv(src, **kw) -> pd.DataFrame:
    if has_module("pyarrow") and "nrows" not in kw:
        try:
            return pd.read_csv(src, engine="pyarrow", **kw)
        except Exception:
            if hasattr(src, "seek"):
                src.seek(0)
    return pd.read_csv(src, low_memory=False, **kw)


def _iter_tables(raw: Path, must_have: str):
    """Yield (name, DataFrame) for every CSV/parquet under raw (or inside zips)
    whose header contains column `must_have`."""
    files = sorted(p for p in raw.rglob("*") if p.is_file())
    data_files = [p for p in files if p.suffix.lower() in {".csv", ".parquet"}]
    if not data_files:  # read straight from zips
        for z in (p for p in files if p.suffix.lower() == ".zip"):
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if not m.lower().endswith(".csv"):
                        continue
                    with zf.open(m) as fh:
                        head = pd.read_csv(fh, nrows=0)
                    if must_have not in [c.strip() for c in head.columns]:
                        continue
                    with zf.open(m) as fh:
                        yield m, _read_csv(io.BytesIO(fh.read()))
        return
    for p in data_files:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
            if must_have in df.columns:
                yield p.name, df
            continue
        head = pd.read_csv(p, nrows=0)
        if must_have in [c.strip() for c in head.columns]:
            yield p.name, _read_csv(p)


# ---------------------------------------------------------------- NetFlow v3
NF_RENAME = {
    "IPV4_SRC_ADDR": "src_ip", "IPV4_DST_ADDR": "dst_ip",
    "L4_SRC_PORT": "src_port", "L4_DST_PORT": "dst_port", "PROTOCOL": "protocol",
    "FLOW_START_MILLISECONDS": "t_start", "FLOW_END_MILLISECONDS": "t_end",
    "IN_BYTES": "bytes_fwd", "OUT_BYTES": "bytes_bwd", "IN_PKTS": "pkts_fwd", "OUT_PKTS": "pkts_bwd",
    "FLOW_DURATION_MILLISECONDS": "duration_ms", "Label": "label", "Attack": "attack",
}
NF_NOT_FEATURE = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "FLOW_START_MILLISECONDS",
                  "FLOW_END_MILLISECONDS", "Label", "Attack", "Dataset"}


def load_netflow_v3(raw: Path, max_rows: int | None = None) -> pd.DataFrame:
    parts = []
    for name, df in _iter_tables(raw, "IPV4_SRC_ADDR"):
        LOG.info("  read %s: %d rows", name, len(df))
        parts.append(df)
        if max_rows and sum(len(p) for p in parts) >= max_rows:
            break
    if not parts:
        raise FileNotFoundError(f"no NetFlow v3 CSV/parquet with IPV4_SRC_ADDR under {raw}")
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    if max_rows:
        df = df.iloc[:max_rows]
    missing = [c for c in ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label") if c not in df.columns]
    if missing:
        raise ValueError(f"NetFlow v3 columns missing: {missing}")

    feats = {}
    for c in df.columns:
        if c in NF_NOT_FEATURE or c.startswith(("FLOW_START", "FLOW_END")):
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().mean() > 0.99:
            feats["f_" + c.lower()] = col.fillna(0).astype("float32")

    out = pd.DataFrame({
        "t_start": pd.to_numeric(df["FLOW_START_MILLISECONDS"], errors="coerce").astype("int64"),
        "t_end": pd.to_numeric(df["FLOW_END_MILLISECONDS"], errors="coerce").astype("int64"),
        "src_ip": df["IPV4_SRC_ADDR"].astype(str),
        "dst_ip": df["IPV4_DST_ADDR"].astype(str),
        "src_port": pd.to_numeric(df["L4_SRC_PORT"], errors="coerce").fillna(0).astype("int32"),
        "dst_port": pd.to_numeric(df["L4_DST_PORT"], errors="coerce").fillna(0).astype("int32"),
        "protocol": pd.to_numeric(df["PROTOCOL"], errors="coerce").fillna(0).astype("int16"),
        "bytes_fwd": pd.to_numeric(df.get("IN_BYTES", 0), errors="coerce"),
        "bytes_bwd": pd.to_numeric(df.get("OUT_BYTES", 0), errors="coerce"),
        "pkts_fwd": pd.to_numeric(df.get("IN_PKTS", 0), errors="coerce"),
        "pkts_bwd": pd.to_numeric(df.get("OUT_PKTS", 0), errors="coerce"),
        "duration_ms": pd.to_numeric(df.get("FLOW_DURATION_MILLISECONDS", 0), errors="coerce"),
        "label": pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype("int8"),
        "attack": normalize_attack(df["Attack"]) if "Attack" in df.columns else "Unknown",
    })
    for c in ("bytes_fwd", "bytes_bwd", "pkts_fwd", "pkts_bwd", "duration_ms"):
        out[c] = out[c].fillna(0).astype("float32")
    out.loc[out["label"] == 0, "attack"] = "Benign"
    out.loc[(out["label"] == 1) & (out["attack"] == "Benign"), "attack"] = "Unknown"
    out["attack_id"] = ""
    out["role"] = ""
    return pd.concat([out, pd.DataFrame(feats)], axis=1)


# ---------------------------------------------------------------- CIDDS-001
PROTO = {"TCP": 6, "UDP": 17, "ICMP": 1, "IGMP": 2, "GRE": 47}
FLAGS = "UAPRSF"


def _parse_bytes(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    mult = np.where(s.str.endswith("G"), 1e9, np.where(s.str.endswith("M"), 1e6,
                    np.where(s.str.endswith("K"), 1e3, 1.0)))
    num = pd.to_numeric(s.str.replace(r"[ GMK]", "", regex=True), errors="coerce").fillna(0)
    return (num * mult).astype("float32")


def _to_ms(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts.astype(str).str.strip(), format="mixed")
    return ((t - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)).astype("int64")


def load_cidds(raw: Path, environment: str = "openstack", weeks=(1, 2, 3, 4),
               drop_victim_replies: bool = True, max_rows: int | None = None) -> pd.DataFrame:
    want_env = {"openstack": ("internal",), "external": ("external",), "both": ("internal", "external")}[environment]
    parts = []
    for name, df in _iter_tables(raw, "Date first seen"):
        low = Path(name).name.lower()
        if not any(e in low for e in want_env):
            continue
        if weeks and not any(f"week{w}" in low for w in weeks):
            continue
        LOG.info("  read %s: %d rows", name, len(df))
        parts.append(df)
        if max_rows and sum(len(p) for p in parts) >= max_rows:
            break
    if not parts:
        raise FileNotFoundError(f"no CIDDS CSV (env={environment}, weeks={weeks}) under {raw}")
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    if max_rows:
        df = df.iloc[:max_rows]

    t_start = _to_ms(df["Date first seen"])
    dur_s = pd.to_numeric(df["Duration"], errors="coerce").fillna(0).astype("float64")
    attack_type = df["attackType"].astype(str).str.strip()
    attack_id = df["attackID"].astype(str).str.strip().replace({"---": ""})
    role = df["class"].astype(str).str.strip()
    is_attack = ~attack_type.str.lower().isin(BENIGN_NAMES)
    flags = df["Flags"].astype(str).str.strip().str.pad(6, side="right", fillchar=".")
    proto = df["Proto"].astype(str).str.strip().str.upper()

    out = pd.DataFrame({
        "t_start": t_start,
        "t_end": t_start + (dur_s * 1000).round().astype("int64"),
        "src_ip": df["Src IP Addr"].astype(str).str.strip(),
        "dst_ip": df["Dst IP Addr"].astype(str).str.strip(),
        "src_port": pd.to_numeric(df["Src Pt"], errors="coerce").fillna(0).astype("int32"),
        "dst_port": pd.to_numeric(df["Dst Pt"], errors="coerce").fillna(0).astype("int32"),
        "protocol": proto.map(PROTO).fillna(0).astype("int16"),
        "bytes_fwd": _parse_bytes(df["Bytes"]),
        "bytes_bwd": np.float32(0),
        "pkts_fwd": pd.to_numeric(df["Packets"], errors="coerce").fillna(0).astype("float32"),
        "pkts_bwd": np.float32(0),
        "duration_ms": (dur_s * 1000).astype("float32"),
        "label": is_attack.astype("int8"),
        "attack": normalize_attack(attack_type.where(is_attack, "Benign")),
        "attack_id": attack_id.where(is_attack, ""),
        "role": role,
    })
    out["f_duration"] = out["duration_ms"]
    out["f_bytes"] = out["bytes_fwd"]
    out["f_packets"] = out["pkts_fwd"]
    out["f_bytes_per_pkt"] = (out["bytes_fwd"] / out["pkts_fwd"].clip(lower=1)).astype("float32")
    out["f_dst_port"] = out["dst_port"].astype("float32")
    out["f_protocol"] = out["protocol"].astype("float32")
    out["f_tos"] = pd.to_numeric(df.get("Tos", 0), errors="coerce").fillna(0).astype("float32")
    for i, fl in enumerate(FLAGS):
        out[f"f_flag_{fl}"] = (flags.str[i] != ".").astype("float32")

    if drop_victim_replies:
        # CIDDS is unidirectional: replies victim->attacker carry the attack id too.
        # NetFlow v3 records are initiator->responder, so drop replies to match.
        reply = (out["label"] == 1) & (out["role"].str.lower() == "victim")
        LOG.info("  dropping %d victim->attacker reply flows", int(reply.sum()))
        out = out.loc[~reply].reset_index(drop=True)
    return out


def load_dataset_raw(cfg: dict, name: str, raw: Path) -> pd.DataFrame:
    reg = cfg["registry"][name]
    max_rows = cfg["data"].get("max_rows")
    if reg["loader"] == "netflow_v3":
        df = load_netflow_v3(raw, max_rows)
    elif reg["loader"] == "cidds":
        c = cfg["data"]["cidds"]
        df = load_cidds(raw, c["environment"], c["weeks"], c["drop_victim_replies"], max_rows)
    else:
        raise ValueError(f"unknown loader {reg['loader']}")
    df = df[df["t_end"] >= df["t_start"]].sort_values("t_start", kind="mergesort").reset_index(drop=True)
    df.insert(0, "flow_id", np.arange(len(df), dtype=np.int64))
    return df
