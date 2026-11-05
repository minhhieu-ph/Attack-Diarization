"""Write synthetic raw data for the smoke test (use with --config configs/smoke.yaml)."""
from _cli import base_parser, parse

from attackdiar.config import dataset_dir
from attackdiar.data.synthetic import make_synthetic_cidds, make_synthetic_netflow
from attackdiar.utils import LOG

if __name__ == "__main__":
    args, cfg = parse(base_parser(__doc__))
    d = dataset_dir(cfg)
    make_synthetic_netflow(d / "raw" / "synth_netflow", d / "schedules" / "synth_netflow.csv")
    make_synthetic_cidds(d / "raw" / "synth_cidds")
    LOG.info("synthetic raw data written under %s", d)
