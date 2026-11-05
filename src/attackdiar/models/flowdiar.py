"""FlowDiar: episode-level model on host x time-bin sequences.

Core (always on):
  temporal encoder : Transformer over the W bins of each host in a window
  graph encoder    : directed message passing on the window's host graph
                     (edges = aggregated flows src->dst), broadcast back to every bin
  heads            : per (host, bin) class (benign + attacks) and role (none/attacker/victim)
Decoding (models/decode.py) turns probabilities into episodes.

Ablations (E6): use_graph=False, use_temporal=False (per-bin MLP), decoder=sessionize.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baselines.torch_flow import scatter_mean
from ..data.features import class_weights
from ..data.hostbins import FEATURES, IGNORE, HostBins, WindowIndex
from ..metrics.episode import episode_metrics
from ..utils import LOG, amp_dtype, resolve_device, set_seed, setup_torch
from .decode import decode_hysteresis, decode_sessionize, flow_labels_from_rows


class FlowDiarNet(nn.Module):
    def __init__(self, d_in, n_classes, window_bins, d_model=128, n_heads=4, n_layers_temporal=2,
                 n_layers_graph=2, dropout=0.1, use_graph=True, use_temporal=True, **_):
        super().__init__()
        self.use_graph, self.use_temporal = use_graph, use_temporal
        self.inp = nn.Linear(d_in + 1, d_model)
        self.pos = nn.Embedding(window_bins, d_model)
        if use_temporal:
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, n_layers_temporal, enable_nested_tensor=False)
            layer2 = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
            self.post = nn.TransformerEncoder(layer2, 1, enable_nested_tensor=False)
        else:
            self.temporal = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            self.post = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        if use_graph:
            self.edge_proj = nn.Sequential(nn.Linear(2, d_model), nn.ReLU())
            self.gl = nn.ModuleList([nn.Linear(3 * d_model, d_model) for _ in range(n_layers_graph)])
            self.gfuse = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head_cls = nn.Linear(d_model, n_classes)
        self.head_role = nn.Linear(d_model, 3)

    def forward(self, x, active, edge_index, edge_attr):
        H, W, _ = x.shape
        z = self.inp(torch.cat([x, active.unsqueeze(-1)], -1)) + self.pos.weight[:W].unsqueeze(0)
        z = self.temporal(z)
        if self.use_graph:
            h = z.mean(1)
            if edge_index.size(1):
                e = self.edge_proj(edge_attr).to(h.dtype)
                src, dst = edge_index[0], edge_index[1]
                for lin in self.gl:
                    m_out = scatter_mean(h[dst] + e, src, H)
                    m_in = scatter_mean(h[src] + e, dst, H)
                    h = h + self.drop(F.relu(lin(torch.cat([h, m_out, m_in], 1))))
            z = z + self.gfuse(h).unsqueeze(1)
        z = self.norm(self.post(z))
        return self.head_cls(z), self.head_role(z)


def collate(hb: HostBins, windows, W: int):
    H = sum(w.n_hosts for w in windows)
    D = hb.X.shape[1]
    X = np.zeros((H, W, D), np.float32)
    act = np.zeros((H, W), np.float32)
    yc = np.full((H, W), IGNORE, np.int64)
    yr = np.full((H, W), IGNORE, np.int64)
    ei, ea, place, off = [], [], [], 0
    for w in windows:
        h = w.h_local + off
        X[h, w.t_local] = hb.X[w.rows]
        act[h, w.t_local] = 1.0
        yc[h, w.t_local] = hb.y_cls[w.rows]
        yr[h, w.t_local] = hb.y_role[w.rows]
        ei.append(w.edge_index + off)
        ea.append(w.edge_attr)
        place.append((w.rows, h, w.t_local))
        off += w.n_hosts
    return {"x": torch.from_numpy(X), "active": torch.from_numpy(act), "y_cls": torch.from_numpy(yc),
            "y_role": torch.from_numpy(yr), "edge_index": torch.from_numpy(np.concatenate(ei, 1)),
            "edge_attr": torch.from_numpy(np.concatenate(ea, 0)), "place": place}


class FlowDiar:
    def __init__(self, cfg: dict, classes: list[str], seed: int, overrides: dict | None = None):
        self.cfg = cfg
        self.p = {**cfg["flowdiar"], **(overrides or {})}
        self.classes, self.seed = classes, seed
        self.device = resolve_device(cfg)
        setup_torch(cfg)
        self.decode_params: dict = {}

    def _net(self):
        return FlowDiarNet(len(FEATURES), len(self.classes), **self.p).to(self.device)

    def _autocast(self):
        return torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu",
                              dtype=amp_dtype(self.cfg), enabled=self.device == "cuda")

    def _to(self, b):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

    # ------------------------------------------------------------ training
    def fit(self, hb_tr: HostBins, hb_va: HostBins, flows_va: pd.DataFrame, gt_va: pd.DataFrame):
        set_seed(self.seed)
        p = self.p
        W, S = p["window_bins"], p["stride_bins"]
        wi = WindowIndex(hb_tr, W, S, p["max_hosts"])
        att = [k for k in range(len(wi)) if wi.has_attack(k)]
        ben = [k for k in range(len(wi)) if not wi.has_attack(k)]
        LOG.info("  FlowDiar: %d windows (%d with attacks), %d host-bins", len(wi), len(att), len(hb_tr.host))
        self.net = self._net()
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        wc = torch.tensor(class_weights(hb_tr.y_cls, len(self.classes)), dtype=torch.float32, device=self.device)
        wr = torch.tensor(class_weights(hb_tr.y_role, 3), dtype=torch.float32, device=self.device)
        wc[wc == 0] = 1.0
        wr[wr == 0] = 1.0
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = -1.0, None, 0
        for ep in range(p["epochs"]):
            self.net.train()
            n_ben = int(len(ben) * p["benign_window_keep"])
            ks = np.array(att + list(rng.choice(ben, n_ben, replace=False) if n_ben else []), dtype=np.int64)
            ks = rng.permutation(ks)
            tot, nb = 0.0, 0
            for i in range(0, len(ks), p["windows_per_batch"]):
                b = self._to(collate(hb_tr, [wi.get(int(k)) for k in ks[i:i + p["windows_per_batch"]]], W))
                with self._autocast():
                    lc, lr = self.net(b["x"], b["active"], b["edge_index"], b["edge_attr"])
                loss = (F.cross_entropy(lc.float().reshape(-1, lc.size(-1)), b["y_cls"].reshape(-1), weight=wc,
                                        ignore_index=IGNORE)
                        + F.cross_entropy(lr.float().reshape(-1, 3), b["y_role"].reshape(-1), weight=wr,
                                          ignore_index=IGNORE))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += float(loss)
                nb += 1
            pc, pr, _ = self.infer(hb_va)
            params, f1 = self.tune_decoder(hb_va, flows_va, gt_va, pc, pr, quick=True)
            LOG.info("  FlowDiar epoch %d loss %.4f val episode-F1 %.4f", ep, tot / max(nb, 1), f1)
            if f1 > best:
                best, best_state, bad = f1, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        pc, pr, _ = self.infer(hb_va)
        self.decode_params, f1 = self.tune_decoder(hb_va, flows_va, gt_va, pc, pr)
        LOG.info("  FlowDiar decoder %s (val episode-F1 %.4f)", self.decode_params, f1)
        return self

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def infer(self, hb: HostBins, mode: str = "offline"):
        """Row probabilities. offline: average over overlapping windows.
        online: each bin is emitted by the earliest window that has it among its last
        `stride` bins; avail_ms = that window's end time (no future data used)."""
        self.net.eval()
        p = self.p
        W, S = p["window_bins"], p["stride_bins"]
        wi = WindowIndex(hb, W, S, p["max_hosts"])
        n, C = len(hb.host), len(self.classes)
        sc, sr, cnt = np.zeros((n, C), np.float32), np.zeros((n, 3), np.float32), np.zeros(n, np.float32)
        avail = np.full(n, np.iinfo(np.int64).max, np.int64)
        bs = p["windows_per_batch"] * 2
        for i in range(0, len(wi), bs):
            ws = [wi.get(k) for k in range(i, min(i + bs, len(wi)))]
            b = collate(hb, ws, W)
            with self._autocast():
                lc, lr = self.net(b["x"].to(self.device), b["active"].to(self.device),
                                  b["edge_index"].to(self.device), b["edge_attr"].to(self.device))
            pc = torch.softmax(lc.float(), -1).cpu().numpy()
            pr = torch.softmax(lr.float(), -1).cpu().numpy()
            for w, (rows, h, t) in zip(ws, b["place"]):
                if mode == "online":
                    end_ms = hb.t0 + (w.b0 + W) * hb.bin_ms
                    m = (t >= W - S) | (w.b0 == 0)
                    m &= avail[rows] == np.iinfo(np.int64).max
                    rows, h, t = rows[m], h[m], t[m]
                    avail[rows] = end_ms
                    sc[rows], sr[rows], cnt[rows] = pc[h, t], pr[h, t], 1
                else:
                    sc[rows] += pc[h, t]
                    sr[rows] += pr[h, t]
                    cnt[rows] += 1
        miss = cnt == 0
        cnt[miss] = 1
        sc /= cnt[:, None]
        sr /= cnt[:, None]
        sc[miss] = np.eye(C, dtype=np.float32)[0]
        sr[miss] = np.eye(3, dtype=np.float32)[0]
        if mode == "online":
            avail[miss] = hb.t0 + (hb.bin[miss] + 1) * hb.bin_ms
            return sc, sr, avail
        return sc, sr, None

    def decode(self, hb, flows, pc, pr, params, avail=None):
        jacc = self.cfg["episodes"]["merge_victim_jaccard"]
        if self.p["decoder"] == "sessionize":
            eps, row_cls = decode_sessionize(hb, flows, pc, pr, self.classes, params["tau_sec"], jacc)
        else:
            eps, row_cls = decode_hysteresis(hb, pc, pr, self.classes, params["high"],
                                             params["high"] * self.p["hysteresis_low_ratio"],
                                             self.p["gap_fill_bins"], self.p["victim_prob"], avail, jacc)
        return eps, flow_labels_from_rows(hb, row_cls, self.classes)

    def tune_decoder(self, hb, flows, gt, pc, pr, quick: bool = False):
        if self.p["decoder"] == "sessionize":
            grid = [{"tau_sec": t} for t in self.p["sessionize_tau_grid_sec"]]
        else:
            grid = [{"high": h} for h in self.p["hysteresis_high_grid"]]
        if quick:
            grid = grid[:1] if self.p["decoder"] == "sessionize" else [{"high": 0.5}]
        best, best_f1 = grid[0], -1.0
        for g in grid:
            eps, _ = self.decode(hb, flows, pc, pr, g)
            f1 = episode_metrics(gt, eps, self.cfg["metrics"])["f1_macro"]
            if f1 > best_f1:
                best, best_f1 = g, f1
        return best, best_f1

    def predict(self, hb: HostBins, flows: pd.DataFrame, mode: str = "offline"):
        pc, pr, avail = self.infer(hb, mode)
        return self.decode(hb, flows, pc, pr, self.decode_params, avail)

    # ------------------------------------------------------------ io
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"p": self.p, "classes": self.classes, "decode": self.decode_params,
                    "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, cfg: dict):
        d = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(cfg, d["classes"], 0, d["p"])
        obj.net = obj._net()
        obj.net.load_state_dict(d["state"])
        obj.decode_params = d["decode"]
        return obj
