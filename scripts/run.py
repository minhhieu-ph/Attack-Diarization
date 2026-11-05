"""Run one or more experiments.

Examples:
  python scripts/run.py --exp E1 E2 E3              # gate (week 1)
  python scripts/run.py --exp E4 --datasets cidds001 --set seeds=[0]
  python scripts/run.py --exp E2 --models B1 B7     # subset of baselines
  python scripts/run.py --exp report
"""
from _cli import base_parser, parse

from attackdiar.experiments import EXPERIMENTS
from attackdiar.report import write_report
from attackdiar.utils import LOG, set_seed

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--exp", nargs="+", required=True, help="E1..E8 or report")
    p.add_argument("--models", nargs="*", default=None, help="baseline subset for E2/E5 (e.g. B1 B7)")
    args, cfg = parse(p)
    if args.models:
        cfg.setdefault("run", {})["models"] = args.models
    set_seed(0)
    for e in args.exp:
        if e.lower() == "report":
            write_report(cfg)
            continue
        LOG.info("==== %s ====", e)
        EXPERIMENTS[e.upper()](cfg)
    if "report" not in [e.lower() for e in args.exp]:
        write_report(cfg)
