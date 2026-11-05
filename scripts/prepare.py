"""Parse raw files, build episode ground truth and splits for each dataset."""
from _cli import base_parser, parse

from attackdiar.data.prepare import prepare_dataset
from attackdiar.utils import LOG

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--force", action="store_true")
    args, cfg = parse(p)
    for name in cfg["data"]["datasets"]:
        try:
            prepare_dataset(cfg, name, args.force)
        except FileNotFoundError as e:
            LOG.error("[%s] %s", name, e)
