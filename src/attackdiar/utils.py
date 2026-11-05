"""Small shared helpers: logging, seeding, IO, device selection."""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

SEP = ";"  # separator for host sets stored as strings


def get_logger(name: str = "attackdiar") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(os.environ.get("ATTACKDIAR_LOG", "INFO"))
        log.propagate = False
    return log


LOG = get_logger()


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if has_module("torch"):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@contextmanager
def timer(msg: str):
    t0 = time.time()
    yield
    LOG.info("%s took %.1fs (RSS %.1f GiB)", msg, time.time() - t0, rss_gib())


# ---------------------------------------------------------------- dataframe IO
def save_df(df: pd.DataFrame, stem: str | Path) -> Path:
    """Save as parquet when pyarrow exists, else pickle. `stem` has no suffix."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if has_module("pyarrow"):
        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    else:
        path = stem.with_suffix(".pkl")
        df.to_pickle(path)
    return path


def load_df(stem: str | Path) -> pd.DataFrame:
    stem = Path(stem)
    for suf in (".parquet", ".pkl"):
        p = stem.with_suffix(suf)
        if p.exists():
            return pd.read_parquet(p) if suf == ".parquet" else pd.read_pickle(p)
    raise FileNotFoundError(f"no parquet/pkl for {stem}")


def df_exists(stem: str | Path) -> bool:
    stem = Path(stem)
    return stem.with_suffix(".parquet").exists() or stem.with_suffix(".pkl").exists()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- host sets
def to_set(s) -> frozenset:
    if isinstance(s, (set, frozenset)):
        return frozenset(s)
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return frozenset()
    return frozenset(str(s).split(SEP))


def join_set(values) -> str:
    return SEP.join(sorted({str(v) for v in values}))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------- resources
def rss_gib() -> float:
    """Resident memory of this process in GiB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


def check_memory(cfg: dict, where: str = "") -> None:
    lim = cfg.get("compute", {}).get("ram_limit_gib")
    used = rss_gib()
    if lim and used == used and used > 0.8 * lim:
        LOG.warning("RAM %.0f GiB of %.0f GiB limit (%s) - reduce data.max_rows or "
                    "flow_models.max_train_per_class to avoid an OOM kill", used, lim, where)
    else:
        LOG.debug("RAM %.1f GiB (%s)", used, where)


def setup_threads(cfg: dict) -> None:
    """Cap every thread pool to compute.n_jobs (containers often report the host's cores)."""
    n = str(int(cfg.get("compute", {}).get("n_jobs", 8)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ.setdefault(var, n)
    if has_module("pyarrow"):
        import pyarrow

        pyarrow.set_cpu_count(int(n))
        pyarrow.set_io_thread_count(min(int(n), 8))


# ---------------------------------------------------------------- device
def resolve_device(cfg: dict) -> str:
    want = cfg.get("compute", {}).get("device", "auto")
    if not has_module("torch"):
        return "cpu"
    import torch

    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        LOG.warning("CUDA requested but not available -> CPU")
        return "cpu"
    return want


def setup_torch(cfg: dict) -> None:
    if not has_module("torch"):
        return
    import torch

    if cfg.get("compute", {}).get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(min(int(cfg.get("compute", {}).get("n_jobs", 8)), os.cpu_count() or 8))


def amp_dtype(cfg: dict):
    import torch

    name = cfg.get("compute", {}).get("amp_dtype", "bfloat16")
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]
