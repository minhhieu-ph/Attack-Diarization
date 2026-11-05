"""CPU-only end-to-end check of data -> GT -> splits -> B1/B7 -> metrics -> stress."""
import numpy as np

from attackdiar.config import load_config
from attackdiar.data.prepare import Dataset, prepare_dataset
from attackdiar.data.stress import PERTURB
from attackdiar.data.synthetic import make_synthetic_cidds, make_synthetic_netflow
from attackdiar.experiments import e1_ground_truth_stats, e2_baselines
from attackdiar.utils import read_jsonl, to_set


def test_end_to_end(tmp_path):
    cfg = load_config(["configs/smoke.yaml"], [f"project.output_dir={tmp_path / 'out'}",
                                               f"project.dataset_dir={tmp_path / 'data'}", "seeds=[0]"])
    d = tmp_path / "data"
    make_synthetic_netflow(d / "raw" / "synth_netflow", d / "schedules" / "synth_netflow.csv", hours=1.5, n_episodes=16)
    make_synthetic_cidds(d / "raw" / "synth_cidds", hours=1.5, n_episodes=16)
    for n in cfg["data"]["datasets"]:
        prepare_dataset(cfg, n)
    ds = Dataset(cfg, "synth_netflow")
    assert ds.meta["gt_source"] == "schedule" and Dataset(cfg, "synth_cidds").meta["gt_source"] == "attack_id"

    rows = e1_ground_truth_stats(cfg)
    assert all(r["n_episodes"] > 5 for r in rows)

    cfg["run"] = {"models": ["B1", "B7"]}
    e2_baselines(cfg)
    res = read_jsonl(tmp_path / "out" / "results" / "E2.jsonl")
    assert {r["model"] for r in res} == {"B1", "B7"}
    assert all(0 <= r["f1_macro"] <= 1 for r in res)

    te, gte = ds.part("temporal", "test")
    f2, g2 = PERTURB["slow"](te, gte, 5)
    assert (g2["t_end"] - g2["t_start"]).sum() > (gte["t_end"] - gte["t_start"]).sum()
    f3, g3 = PERTURB["split_sources"](te, gte, 3)
    assert all(len(to_set(a)) >= 3 for a in g3["attackers"])
    f4, g4 = PERTURB["interleave"](te, gte, 1.0)
    assert len(f4) > len(te) and len(g4) == len(gte)
