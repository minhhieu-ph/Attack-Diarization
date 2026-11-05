"""PyTorch per-flow classifiers.

B4  ftt : FT-Transformer on the flow's own features (no context).
B5  gnn : E-GraphSAGE-style edge classifier on host graphs of consecutive time chunks.
B6  tgn : TGN-lite, per-host GRU memory updated in time order (dynamic graph).
All are pure PyTorch (no PyG/DGL) so they install with a single wheel.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.features import Standardizer, class_weights, encode, feature_columns, subsample_per_class
from ..metrics.flow import flow_macro_f1
from ..utils import LOG, amp_dtype, resolve_device, set_seed, setup_torch


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of rows of `src` grouped by `index` (accumulated in float32)."""
    out = torch.zeros(n, src.size(1), device=src.device, dtype=torch.float32)
    out.index_add_(0, index, src.float())
    cnt = torch.zeros(n, device=src.device, dtype=torch.float32)
    cnt.index_add_(0, index, torch.ones(index.size(0), device=src.device, dtype=torch.float32))
    return (out / cnt.clamp(min=1).unsqueeze(1)).to(src.dtype)


class FTTransformer(nn.Module):
    def __init__(self, n_feat, n_classes, d_model=64, n_layers=3, n_heads=4, dropout=0.1, **_):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_feat, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, n_classes))

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b
        tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
        return self.head(self.enc(tok)[:, 0])


class EdgeGNN(nn.Module):
    def __init__(self, n_feat, n_classes, d_model=128, n_layers=2, dropout=0.1, **_):
        super().__init__()
        self.edge_in = nn.Sequential(nn.Linear(n_feat, d_model), nn.ReLU())
        self.layers = nn.ModuleList([nn.Linear(3 * d_model, d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_classes))

    def forward(self, x, src, dst, n_nodes):
        e = self.edge_in(x)
        h = torch.zeros(n_nodes, e.size(1), device=x.device, dtype=e.dtype)
        for lin in self.layers:
            m_out = scatter_mean(e + h[dst], src, n_nodes)
            m_in = scatter_mean(e + h[src], dst, n_nodes)
            h = self.drop(F.relu(lin(torch.cat([h, m_out, m_in], 1))))
        return self.out(torch.cat([h[src], h[dst], e], 1))


class TGNLite(nn.Module):
    def __init__(self, n_feat, n_classes, d_model=100, dropout=0.1, **_):
        super().__init__()
        self.x_enc = nn.Sequential(nn.Linear(n_feat, d_model), nn.ReLU())
        self.msg = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU())
        self.gru = nn.GRUCell(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_classes))
        self.d = d_model

    def classify(self, mem, x, src, dst):
        xe = self.x_enc(x)
        return self.cls(self.drop(torch.cat([mem[src].to(xe.dtype), mem[dst].to(xe.dtype), xe], 1)))

    def update(self, mem, x, src, dst):
        xe = self.x_enc(x)
        ms, md = mem[src].to(xe.dtype), mem[dst].to(xe.dtype)
        m = torch.cat([self.msg(torch.cat([xe, md], 1)), self.msg(torch.cat([xe, ms], 1))], 0)
        nodes = torch.cat([src, dst], 0)
        agg = scatter_mean(m, nodes, mem.size(0))
        uniq = torch.unique(nodes)
        new = self.gru(agg[uniq].float(), mem[uniq])
        return mem.index_copy(0, uniq, new.to(mem.dtype))


ARCH = {"ftt": FTTransformer, "gnn": EdgeGNN, "tgn": TGNLite}


class TorchFlowModel:
    def __init__(self, kind, classes, params, cfg, seed):
        self.kind, self.classes, self.p, self.cfg, self.seed = kind, classes, dict(params), cfg, seed
        self.device = resolve_device(cfg)
        setup_torch(cfg)

    # ------------------------------------------------------------ data
    def _arrays(self, flows: pd.DataFrame, fit: bool = False):
        X = flows[self.cols].to_numpy(np.float32)
        if fit:
            self.scaler = Standardizer().fit(X)
        X = self.scaler.transform(X)
        y = encode(flows["attack"], self.classes)
        return X, y

    def _graph_ids(self, flows: pd.DataFrame):
        ips = np.concatenate([flows["src_ip"].to_numpy(), flows["dst_ip"].to_numpy()])
        codes, uniq = pd.factorize(ips)
        n = len(flows)
        return codes[:n], codes[n:], len(uniq)

    # ------------------------------------------------------------ train
    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        set_seed(self.seed)
        self.cols = feature_columns(train)
        X, y = self._arrays(train, fit=True)
        Xv, yv = self._arrays(val)
        C = len(self.classes)
        self.model = ARCH[self.kind](X.shape[1], C, **self.p).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.p["lr"], weight_decay=self.p.get("weight_decay", 0))
        w = torch.tensor(class_weights(y, C), dtype=torch.float32, device=self.device)
        best, best_state, bad = -1.0, None, 0
        for ep in range(self.p["epochs"]):
            self.model.train()
            loss = getattr(self, f"_epoch_{self.kind}")(train, X, y, opt, w)
            f1 = flow_macro_f1(yv, self._predict(val, Xv).argmax(1)) if len(yv) else 0.0
            LOG.info("  [%s] epoch %d loss %.4f val flowF1 %.4f", self.kind, ep, loss, f1)
            if f1 > best:
                best, best_state, bad = f1, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.p["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _step(self, opt, logits, target, w):
        m = target >= 0
        if not m.any():
            return 0.0
        loss = F.cross_entropy(logits[m].float(), target[m], weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        return float(loss)

    def _autocast(self):
        return torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu",
                              dtype=amp_dtype(self.cfg), enabled=self.device == "cuda")

    def _epoch_ftt(self, train, X, y, opt, w):
        keep = subsample_per_class(y, self.cfg["flow_models"]["max_train_per_class"], self.seed)
        rng = np.random.default_rng(self.seed + 17)
        keep = rng.permutation(keep)
        bs, tot, n = self.p["batch_size"], 0.0, 0
        for i in range(0, len(keep), bs):
            b = keep[i:i + bs]
            xb = torch.from_numpy(X[b]).to(self.device, non_blocking=True)
            yb = torch.from_numpy(y[b]).to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(xb)
            tot += self._step(opt, logits, yb, w)
            n += 1
        return tot / max(n, 1)

    def _chunks(self, n_rows):
        cs = self.p["chunk_flows"]
        return [(i, min(i + cs, n_rows)) for i in range(0, n_rows, cs)]

    def _epoch_gnn(self, train, X, y, opt, w):
        src, dst, _ = self._graph_ids(train)
        chunks = self._chunks(len(train))
        rng = np.random.default_rng(self.seed + 31)
        tot = 0.0
        for k in rng.permutation(len(chunks)):
            a, b = chunks[k]
            s_c, d_c = src[a:b], dst[a:b]
            nodes, inv = np.unique(np.concatenate([s_c, d_c]), return_inverse=True)
            s_l = torch.from_numpy(inv[: b - a]).to(self.device)
            d_l = torch.from_numpy(inv[b - a:]).to(self.device)
            xb = torch.from_numpy(X[a:b]).to(self.device)
            yb = torch.from_numpy(y[a:b]).to(self.device)
            with self._autocast():
                logits = self.model(xb, s_l, d_l, len(nodes))
            tot += self._step(opt, logits, yb, w)
        return tot / max(len(chunks), 1)

    def _epoch_tgn(self, train, X, y, opt, w):
        src, dst, n_nodes = self._graph_ids(train)
        mem = torch.zeros(n_nodes, self.model.d, device=self.device)
        pending, tot, n = None, 0.0, 0
        bs = self.p["batch_flows"]
        for i in range(0, len(train), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(self.device)
            sb = torch.from_numpy(src[i:i + bs]).to(self.device)
            db = torch.from_numpy(dst[i:i + bs]).to(self.device)
            yb = torch.from_numpy(y[i:i + bs]).to(self.device)
            with self._autocast():
                mem_u = self.model.update(mem, *pending) if pending is not None else mem
                logits = self.model.classify(mem_u, xb, sb, db)
            tot += self._step(opt, logits, yb, w)
            n += 1
            mem = mem_u.detach().float()
            pending = (xb, sb, db)
        return tot / max(n, 1)

    # ------------------------------------------------------------ predict
    @torch.no_grad()
    def _predict(self, flows: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = np.zeros((len(flows), len(self.classes)), np.float32)
        if len(flows) == 0:
            return out
        if self.kind == "ftt":
            bs = self.p["batch_size"] * 4
            for i in range(0, len(X), bs):
                with self._autocast():
                    lo = self.model(torch.from_numpy(X[i:i + bs]).to(self.device))
                out[i:i + bs] = torch.softmax(lo.float(), 1).cpu().numpy()
        elif self.kind == "gnn":
            src, dst, _ = self._graph_ids(flows)
            for a, b in self._chunks(len(flows)):
                nodes, inv = np.unique(np.concatenate([src[a:b], dst[a:b]]), return_inverse=True)
                with self._autocast():
                    lo = self.model(torch.from_numpy(X[a:b]).to(self.device),
                                    torch.from_numpy(inv[: b - a]).to(self.device),
                                    torch.from_numpy(inv[b - a:]).to(self.device), len(nodes))
                out[a:b] = torch.softmax(lo.float(), 1).cpu().numpy()
        else:
            src, dst, n_nodes = self._graph_ids(flows)
            mem = torch.zeros(n_nodes, self.model.d, device=self.device)
            bs, pending = self.p["batch_flows"], None
            for i in range(0, len(flows), bs):
                xb = torch.from_numpy(X[i:i + bs]).to(self.device)
                sb = torch.from_numpy(src[i:i + bs]).to(self.device)
                db = torch.from_numpy(dst[i:i + bs]).to(self.device)
                with self._autocast():
                    if pending is not None:
                        mem = self.model.update(mem, *pending).float()
                    lo = self.model.classify(mem, xb, sb, db)
                out[i:i + bs] = torch.softmax(lo.float(), 1).cpu().numpy()
                pending = (xb, sb, db)
        return out

    def predict_proba(self, flows: pd.DataFrame) -> np.ndarray:
        X, _ = self._arrays(flows)
        return self._predict(flows, X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": self.kind, "classes": self.classes, "params": self.p, "cols": self.cols,
                    "scaler": self.scaler.__dict__, "state": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, cfg: dict):
        d = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(d["kind"], d["classes"], d["params"], cfg, 0)
        obj.cols = d["cols"]
        obj.scaler = Standardizer()
        obj.scaler.__dict__.update(d["scaler"])
        obj.model = ARCH[obj.kind](len(obj.cols), len(obj.classes), **obj.p).to(obj.device)
        obj.model.load_state_dict(d["state"])
        return obj
