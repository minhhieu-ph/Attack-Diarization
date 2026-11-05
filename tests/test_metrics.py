import numpy as np
import pandas as pd

from attackdiar.metrics.episode import episode_error_rate, episode_metrics, match_episodes
from attackdiar.metrics.stats import paired_permutation_test

M = {"iou_threshold": 0.5, "collar_sec": 0, "attribution_jaccard": 0.5, "minute_sec": 60}


def eps(rows):
    return pd.DataFrame(rows, columns=["attack", "t_start", "t_end", "attackers", "victims"]).assign(
        t_alert=lambda d: d["t_start"])


def test_perfect_prediction():
    g = eps([("dos", 0, 10_000, "a", "v"), ("scan", 20_000, 30_000, "b", "x;y")])
    m = episode_metrics(g, g.copy(), M)
    assert m["f1_macro"] == 1.0 and m["attr_acc_macro"] == 1.0
    assert m["eer"] == 0.0 and m["ttd_median_s"] == 0.0


def test_one_long_prediction_matches_only_one_gt():
    g = eps([("dos", 0, 10_000, "a", "v"), ("dos", 12_000, 22_000, "a", "v")])
    p = eps([("dos", 0, 22_000, "a", "v")])
    pairs = match_episodes(g, p, 0.4, 0)
    assert len(pairs) == 1
    m = episode_metrics(g, p, {**M, "iou_threshold": 0.4})
    assert m["recall_macro"] == 0.5 and m["precision_macro"] == 1.0


def test_wrong_class_and_attribution():
    g = eps([("dos", 0, 10_000, "a", "v")])
    p = eps([("scan", 0, 10_000, "a", "v")])
    m = episode_metrics(g, p, M)
    assert m["f1_macro"] == 0.0
    assert abs(episode_error_rate(g, p, 0) - 1.0) < 1e-6  # all time confused
    p2 = eps([("dos", 0, 10_000, "z", "v")])
    m2 = episode_metrics(g, p2, M)
    assert m2["f1_macro"] == 1.0 and m2["attr_acc_macro"] == 0.0


def test_ttd_and_false_alarm():
    g = eps([("dos", 0, 60_000, "a", "v")])
    p = eps([("dos", 5_000, 60_000, "a", "v"), ("dos", 200_000, 210_000, "q", "v")])
    flows = pd.DataFrame({"t_start": np.arange(0, 3_600_000, 1000)})
    m = episode_metrics(g, p, M, test_flows=flows)
    assert m["ttd_median_s"] == 5.0
    assert m["false_alarm_episodes"] == 1
    assert abs(m["benign_hours"] - 58 / 60) < 1e-9  # GT end is inclusive -> touches minute 1


def test_permutation():
    assert paired_permutation_test([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]) <= 0.07
    assert paired_permutation_test([0.5, 0.4], [0.5, 0.4]) == 1.0
