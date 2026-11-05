"""Host x time-bin representation used by FlowDiar.

Each active (host, bin) pair becomes one row with D aggregated features of the flows
the host sent (out) and received (in) in that bin. This keeps the model size
independent of the number of flows (DDoS bins with millions of flows cost the same
as quiet ones). Edges (src, dst, bin) carry flow counts and bytes for the graph.

Labels per row come from GT *episodes* (not from flow labels):
    cls  = attack class index (0 = benign)
    role = 0 none, 1 attacker, 2 victim
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils import LOG, to_set

FEATURES = ["out_n", "out_bytes_f", "out_bytes_b", "out_pkts_f", "out_pkts_b", "out_dur", "out_peers",
            "out_dports", "out_tcp", "out_udp", "out_icmp", "in_n", "in_bytes_f", "in_pkts_f", "in_peers", "in_dports"]
LOG_FEATS = {"out_n", "out_bytes_f", "out_bytes_b", "out_pkts_f", "out_pkts_b", "out_dur", "out_peers",
             "out_dports", "in_n", "in_bytes_f", "in_pkts_f", "in_peers", "in_dports"}
IGNORE = -100


@dataclass
class HostBins:
    t0: int
    bin_ms: int
    hosts: np.ndarray        # host code -> ip
    host: np.ndarray         # row -> host code   (rows sorted by bin, then host)
    bin: np.ndarray          # row -> bin
    X: np.ndarray            # (rows, D) float32
    y_cls: np.ndarray        # (rows,) int64
    y_role: np.ndarray       # (rows,) int64
    e_src: np.ndarray        # edges sorted by bin
    e_dst: np.ndarray
    e_bin: np.ndarray
    e_cnt: np.ndarray
    e_bytes: np.ndarray
    flow_host: np.ndarray    # flow -> src host code
    flow_bin: np.ndarray     # flow -> bin

    @property
    def n_bins(self) -> int:
        return int(self.bin.max()) + 1 if len(self.bin) else 0

    def row_lookup(self) -> pd.Series:
        """(host, bin) key -> row index."""
        key = self.host.astype(np.int64) * (self.n_bins + 1) + self.bin
        return pd.Series(np.arange(len(key)), index=key)

    def flow_rows(self) -> np.ndarray:
        key = self.flow_host.astype(np.int64) * (self.n_bins + 1) + self.flow_bin
        return self.row_lookup().reindex(key).fillna(-1).astype(np.int64).to_numpy()


def build_hostbins(flows: pd.DataFrame, bin_sec: float, episodes: pd.DataFrame | None = None,
                   classes: list[str] | None = None) -> HostBins:
    bin_ms = int(bin_sec * 1000)
    t0 = int(flows["t_start"].min()) // bin_ms * bin_ms if len(flows) else 0
    b = ((flows["t_start"].to_numpy() - t0) // bin_ms).astype(np.int64)
    ips = np.concatenate([flows["src_ip"].to_numpy(), flows["dst_ip"].to_numpy()])
    codes, hosts = pd.factorize(ips)
    n = len(flows)
    s, d = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    proto = flows["protocol"].to_numpy()
    out = pd.DataFrame({
        "host": s, "bin": b, "bytes_f": flows["bytes_fwd"].to_numpy(), "bytes_b": flows["bytes_bwd"].to_numpy(),
        "pkts_f": flows["pkts_fwd"].to_numpy(), "pkts_b": flows["pkts_bwd"].to_numpy(),
        "dur": flows["duration_ms"].to_numpy(), "peer": d, "dport": flows["dst_port"].to_numpy(),
        "tcp": proto == 6, "udp": proto == 17, "icmp": proto == 1})
    go = out.groupby(["host", "bin"], sort=False)
    A = pd.DataFrame({
        "out_n": go.size(), "out_bytes_f": go["bytes_f"].sum(), "out_bytes_b": go["bytes_b"].sum(),
        "out_pkts_f": go["pkts_f"].sum(), "out_pkts_b": go["pkts_b"].sum(), "out_dur": go["dur"].mean(),
        "out_peers": go["peer"].nunique(), "out_dports": go["dport"].nunique(),
        "out_tcp": go["tcp"].mean(), "out_udp": go["udp"].mean(), "out_icmp": go["icmp"].mean()})
    inn = pd.DataFrame({"host": d, "bin": b, "bytes_f": out["bytes_f"], "pkts_f": out["pkts_f"],
                        "peer": s, "dport": out["dport"]})
    gi = inn.groupby(["host", "bin"], sort=False)
    B = pd.DataFrame({"in_n": gi.size(), "in_bytes_f": gi["bytes_f"].sum(), "in_pkts_f": gi["pkts_f"].sum(),
                      "in_peers": gi["peer"].nunique(), "in_dports": gi["dport"].nunique()})
    T = A.join(B, how="outer").fillna(0.0).reset_index().sort_values(["bin", "host"], kind="mergesort")
    for c in LOG_FEATS:
        T[c] = np.log1p(T[c].astype(np.float64))
    X = T[FEATURES].to_numpy(np.float32)
    host_r = T["host"].to_numpy(np.int64)
    bin_r = T["bin"].to_numpy(np.int64)

    e = pd.DataFrame({"src": s, "dst": d, "bin": b, "bytes": out["bytes_f"]})
    ge = e.groupby(["bin", "src", "dst"], sort=True)
    E = pd.DataFrame({"cnt": ge.size(), "bytes": ge["bytes"].sum()}).reset_index()

    y_cls = np.zeros(len(T), np.int64)
    y_role = np.zeros(len(T), np.int64)
    if episodes is not None and classes is not None and len(episodes):
        y_cls, y_role = _labels(episodes, classes, hosts, t0, bin_ms, host_r, bin_r)

    return HostBins(t0=t0, bin_ms=bin_ms, hosts=np.asarray(hosts, dtype=object), host=host_r, bin=bin_r, X=X,
                    y_cls=y_cls, y_role=y_role, e_src=E["src"].to_numpy(np.int64), e_dst=E["dst"].to_numpy(np.int64),
                    e_bin=E["bin"].to_numpy(np.int64), e_cnt=E["cnt"].to_numpy(np.float32),
                    e_bytes=E["bytes"].to_numpy(np.float32), flow_host=s, flow_bin=b)


def _labels(episodes, classes, hosts, t0, bin_ms, host_r, bin_r):
    cidx = {c: i for i, c in enumerate(classes)}
    code = {h: i for i, h in enumerate(hosts)}
    recs = []
    for r in episodes.itertuples(index=False):
        c = cidx.get(r.attack)
        if c is None:
            continue
        bs, be = (r.t_start - t0) // bin_ms, (r.t_end - t0) // bin_ms
        for role, hs in ((1, to_set(r.attackers)), (2, to_set(r.victims))):
            for h in hs:
                if h in code:
                    recs.append((code[h], bs, be, c, role))
    y_cls = np.zeros(len(host_r), np.int64)
    y_role = np.zeros(len(host_r), np.int64)
    if not recs:
        return y_cls, y_role
    iv = pd.DataFrame(recs, columns=["host", "bs", "be", "cls", "role"])
    rows = pd.DataFrame({"host": host_r, "bin": bin_r, "row": np.arange(len(host_r))})
    m = rows.merge(iv, on="host")
    m = m[(m["bin"] >= m["bs"]) & (m["bin"] <= m["be"])].sort_values(["row", "role"])  # attacker wins
    n_conf = int(m.duplicated("row").sum())
    if n_conf:
        LOG.info("  hostbins: %d host-bins belong to >1 episode (first kept)", n_conf)
    m = m.drop_duplicates("row")
    y_cls[m["row"].to_numpy()] = m["cls"].to_numpy()
    y_role[m["row"].to_numpy()] = m["role"].to_numpy()
    return y_cls, y_role


@dataclass
class Window:
    b0: int
    rows: np.ndarray       # global row ids inside the window (after host cap)
    h_local: np.ndarray    # local host index per row
    t_local: np.ndarray    # bin offset per row
    n_hosts: int
    edge_index: np.ndarray  # (2, E) local host ids
    edge_attr: np.ndarray   # (E, 2) log1p count, log1p bytes
    has_attack: bool


class WindowIndex:
    """Slices HostBins into fixed windows of `W` bins with stride `S`."""

    def __init__(self, hb: HostBins, W: int, S: int, max_hosts: int):
        self.hb, self.W, self.S, self.max_hosts = hb, W, S, max_hosts
        nb = hb.n_bins
        last = max(0, nb - W)
        starts = list(range(0, last + 1, S))
        if starts[-1] < last:
            starts.append(last)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.row_off = np.searchsorted(hb.bin, np.arange(nb + W + 1))
        self.edge_off = np.searchsorted(hb.e_bin, np.arange(nb + W + 1))
        att = np.zeros(nb + 1, np.int64)
        np.add.at(att, hb.bin[hb.y_cls > 0], 1)
        self.att_cum = np.r_[0, np.cumsum(att)]

    def __len__(self):
        return len(self.starts)

    def has_attack(self, k: int) -> bool:
        b0 = self.starts[k]
        hi = min(b0 + self.W, len(self.att_cum) - 1)
        return bool(self.att_cum[hi] - self.att_cum[b0] > 0)

    def get(self, k: int) -> Window:
        hb, b0 = self.hb, int(self.starts[k])
        b1 = b0 + self.W
        r0, r1 = self.row_off[min(b0, len(self.row_off) - 1)], self.row_off[min(b1, len(self.row_off) - 1)]
        rows = np.arange(r0, r1)
        hosts = hb.host[rows]
        uniq, inv, cnt = np.unique(hosts, return_inverse=True, return_counts=True)
        if len(uniq) > self.max_hosts:
            keep_h = np.zeros(len(uniq), bool)
            keep_h[np.argsort(-cnt, kind="stable")[: self.max_hosts]] = True
            m = keep_h[inv]
            rows, hosts = rows[m], hosts[m]
            uniq, inv = np.unique(hosts, return_inverse=True)
        e0, e1 = self.edge_off[min(b0, len(self.edge_off) - 1)], self.edge_off[min(b1, len(self.edge_off) - 1)]
        es, ed = hb.e_src[e0:e1], hb.e_dst[e0:e1]
        ls = np.searchsorted(uniq, es)
        ld = np.searchsorted(uniq, ed)
        ok = (ls < len(uniq)) & (ld < len(uniq))
        ok[ok] &= (uniq[ls[ok]] == es[ok]) & (uniq[ld[ok]] == ed[ok])
        if ok.any():
            ed_df = pd.DataFrame({"s": ls[ok], "d": ld[ok], "c": hb.e_cnt[e0:e1][ok], "b": hb.e_bytes[e0:e1][ok]})
            agg = ed_df.groupby(["s", "d"], sort=False).sum().reset_index()
            edge_index = agg[["s", "d"]].to_numpy(np.int64).T
            edge_attr = np.log1p(agg[["c", "b"]].to_numpy(np.float32))
        else:
            edge_index = np.zeros((2, 0), np.int64)
            edge_attr = np.zeros((0, 2), np.float32)
        return Window(b0=b0, rows=rows, h_local=inv.astype(np.int64), t_local=(hb.bin[rows] - b0).astype(np.int64),
                      n_hosts=len(uniq), edge_index=edge_index, edge_attr=edge_attr,
                      has_attack=bool((hb.y_cls[rows] > 0).any()))
