"""Shared argparse helpers for the scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attackdiar.config import load_config  # noqa: E402
from attackdiar.utils import setup_threads  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", action="append", default=[], help="extra YAML (repeatable)")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="override config, e.g. --set flowdiar.lr=1e-4")
    p.add_argument("--datasets", nargs="*", default=None, help="subset of data.datasets")
    return p


def parse(p: argparse.ArgumentParser):
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.datasets:
        cfg["data"]["datasets"] = args.datasets
    setup_threads(cfg)
    return args, cfg
