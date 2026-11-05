"""Download raw datasets into dataset/raw/<name>/.

Examples:
  python scripts/download.py                       # all datasets in config
  python scripts/download.py --datasets cidds001
  python scripts/download.py --datasets nf_ton_iot_v3 --source hf
"""
from _cli import base_parser, parse

from attackdiar.data.download import download_dataset
from attackdiar.utils import LOG

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--source", choices=["direct", "hf"], default="direct")
    p.add_argument("--force", action="store_true")
    args, cfg = parse(p)
    ok = {}
    for name in cfg["data"]["datasets"]:
        if name.startswith("synth_"):
            continue
        ok[name] = download_dataset(cfg, name, args.source, args.force)
    for k, v in ok.items():
        LOG.info("%-24s %s", k, "OK" if v else "NEEDS MANUAL DOWNLOAD (see message above)")
