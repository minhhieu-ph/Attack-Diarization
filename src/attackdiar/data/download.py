"""Download raw datasets into dataset/raw/<name>/.

Direct links (CIDDS-001) are fetched with streaming + resume. The UQ research
data links for NetFlow v3 are landing pages; we try them, look for a direct
file link in the returned HTML, and otherwise print manual instructions.
An optional Hugging Face mirror can be used with source="hf".
"""
from __future__ import annotations

import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from ..config import dataset_dir
from ..utils import LOG, has_module

CHUNK = 1 << 22
UA = {"User-Agent": "Mozilla/5.0 (attackdiar downloader)"}


def _fetch(url: str, dest: Path) -> str:
    """Stream url -> dest (resumes partial files). Returns the content-type."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    headers = dict(UA)
    done = part.stat().st_size if part.exists() else 0
    if done:
        headers["Range"] = f"bytes={done}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        ctype = r.headers.get("Content-Type", "")
        total = r.headers.get("Content-Length")
        mode = "ab" if done and r.status == 206 else "wb"
        LOG.info("GET %s (%s, %s bytes)", url, ctype, total)
        with open(part, mode) as f:
            shutil.copyfileobj(r, f, CHUNK)
    part.rename(dest)
    return ctype


def _looks_like_html(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(512).lower()
    return b"<html" in head or b"<!doctype html" in head


def _find_file_links(html: str, base: str) -> list[str]:
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    out = []
    for link in links:
        if re.search(r"\.(zip|csv|parquet|gz)(\?|$)", link, re.I) or "download" in link.lower():
            out.append(urllib.parse.urljoin(base, link))
    return list(dict.fromkeys(out))


def _extract(path: Path) -> None:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            LOG.info("extracting %s (%d members)", path.name, len(z.namelist()))
            z.extractall(path.parent)


def download_dataset(cfg: dict, name: str, source: str = "direct", force: bool = False) -> bool:
    reg = cfg["registry"][name]
    raw = dataset_dir(cfg) / "raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    have = [p for p in raw.rglob("*") if p.suffix.lower() in {".csv", ".parquet", ".zip"}]
    if have and not force:
        LOG.info("[%s] raw files already present (%d) -> skip", name, len(have))
        return True

    if source == "hf":
        return _download_hf(cfg, name, raw)

    url = reg.get("url")
    if not url:
        LOG.warning("[%s] no url in configs/datasets.yaml", name)
        return False
    fname = Path(urllib.parse.urlparse(url).path).name or f"{name}.bin"
    if "." not in fname:
        fname += ".download"
    dest = raw / fname
    try:
        _fetch(url, dest)
    except Exception as e:  # network/proxy errors
        LOG.error("[%s] download failed: %s", name, e)
        _manual(name, reg, raw)
        return False

    if _looks_like_html(dest):
        html = dest.read_text(errors="ignore")
        dest.unlink()
        for link in _find_file_links(html, url):
            try:
                target = raw / (Path(urllib.parse.urlparse(link).path).name or f"{name}.zip")
                _fetch(link, target)
                if not _looks_like_html(target):
                    _extract(target)
                    return True
                target.unlink()
            except Exception as e:
                LOG.warning("[%s] candidate link failed %s: %s", name, link, e)
        _manual(name, reg, raw)
        return False

    if dest.suffix == ".download":
        new = dest.with_suffix(".zip")
        dest.rename(new)
        dest = new
    _extract(dest)
    return True


def _download_hf(cfg: dict, name: str, raw: Path) -> bool:
    if not has_module("huggingface_hub"):
        LOG.error("pip install huggingface_hub to use the mirror")
        return False
    from huggingface_hub import HfApi, hf_hub_download

    repo = cfg["registry"]["hf_mirror"]["repo_id"]
    pat = cfg["registry"][name].get("hf_pattern", "")
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if pat.lower() in f.lower() and f.endswith((".parquet", ".csv", ".zip"))]
    if not files:
        LOG.error("[%s] no file matching %r in %s", name, pat, repo)
        return False
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(raw))
        LOG.info("[%s] got %s", name, p)
    return True


def _manual(name: str, reg: dict, raw: Path) -> None:
    LOG.warning(
        "[%s] automatic download did not return a data file.\n"
        "  1) Open %s and download the archive for this dataset\n"
        "     (or try: python scripts/download.py --datasets %s --source hf)\n"
        "  2) Put the zip/csv/parquet into %s\n"
        "  3) Run: python scripts/prepare.py --datasets %s",
        name, reg.get("page") or reg.get("url"), name, raw, name,
    )
