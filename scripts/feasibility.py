"""Pilot: download (if needed) -> prepare -> quick H1 gate with oracle ceiling.

Examples:
  python scripts/feasibility.py --config configs/b300.yaml                      # CIDDS-001
  python scripts/feasibility.py --config configs/b300.yaml --flowdiar           # + quick FlowDiar
  python scripts/feasibility.py --config configs/b300.yaml --datasets cidds001 nf_ton_iot_v3
  python scripts/feasibility.py --config configs/b300.yaml --set data.max_rows=5000000   # faster trial
"""
from _cli import base_parser, parse

from attackdiar.data.download import download_dataset
from attackdiar.data.prepare import prepare_dataset
from attackdiar.feasibility import pilot_dataset, write_summary
from attackdiar.utils import LOG, set_seed

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--flowdiar", action="store_true", help="also train a quick FlowDiar (needs torch)")
    p.add_argument("--fd-epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-download", action="store_true")
    args, cfg = parse(p)
    if args.datasets is None:
        cfg["data"]["datasets"] = ["cidds001"]  # direct download + independent ground truth
    cfg["flowdiar"]["epochs"] = args.fd_epochs
    set_seed(args.seed)
    results = []
    for name in cfg["data"]["datasets"]:
        if not args.no_download and not name.startswith("synth_"):
            download_dataset(cfg, name)
        try:
            prepare_dataset(cfg, name)
        except FileNotFoundError as e:
            LOG.error("[%s] %s", name, e)
            continue
        results.append(pilot_dataset(cfg, name, args.seed, args.flowdiar))
    if results:
        print("\n" + write_summary(cfg, results))
