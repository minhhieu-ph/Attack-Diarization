import numpy as np
import pandas as pd

from attackdiar.baselines.postprocess import dbscan_1d, viterbi_per_host
from attackdiar.data.episodes import episodes_from_flow_labels, gt_from_attack_id, sessionize
from attackdiar.data.splits import _move_cut_to_gap
from attackdiar.utils import to_set


def flows(rows):
    return pd.DataFrame(rows, columns=["t_start", "t_end", "src_ip", "dst_ip", "attack"])


def test_sessionize_gap_and_classes():
    f = flows([(0, 10, "a", "v", "dos"), (20, 30, "a", "v", "dos"), (10_000, 10_010, "a", "v", "dos"),
               (5, 6, "a", "w", "scan"), (0, 1, "b", "v", "dos")])
    eps, seg = sessionize(f.t_start, f.t_end, f.src_ip, f.dst_ip, f.attack, gap_ms=1000)
    assert len(eps) == 4  # a/dos split by the 10 s gap, a/scan, b/dos
    assert seg[0] == seg[1] != seg[2]


def test_multi_attacker_merge():
    f = flows([(0, 100, "a1", "v", "ddos"), (5, 90, "a2", "v", "ddos"), (0, 100, "a3", "x", "ddos")])
    eps, fe = episodes_from_flow_labels(f, f.attack.to_numpy(), 1000, 0.5)
    assert len(eps) == 2
    big = eps.loc[eps.attackers.map(lambda s: len(to_set(s))) == 2].iloc[0]
    assert to_set(big.attackers) == {"a1", "a2"} and fe[0] == fe[1] != fe[2]


def test_gt_from_attack_id_uses_attacker_role():
    df = pd.DataFrame({"t_start": [0, 5, 7], "t_end": [1, 6, 8], "src_ip": ["a", "v", "a"], "dst_ip": ["v", "a", "v"],
                       "label": [1, 1, 0], "attack": ["dos", "dos", "Benign"], "attack_id": ["1", "1", ""],
                       "role": ["attacker", "victim", "normal"]})
    eps, fe = gt_from_attack_id(df)
    assert len(eps) == 1 and eps.attackers[0] == "a" and eps.victims[0] == "v"
    assert list(fe) == [0, 0, -1]


def test_viterbi_smooths_isolated_flip():
    n = 9
    f = pd.DataFrame({"src_ip": ["h"] * n, "t_start": np.arange(n)})
    P = np.tile([0.1, 0.9], (n, 1))
    P[4] = [0.6, 0.4]  # one noisy flow inside an attack run
    out = viterbi_per_host(f, P, 0.99)
    assert (out == 1).all()


def test_viterbi_multiple_hosts_lengths():
    f = pd.DataFrame({"src_ip": ["a"] * 5 + ["b"] * 2 + ["c"], "t_start": np.arange(8)})
    P = np.array([[.2, .8]] * 5 + [[.9, .1], [.1, .9]] + [[.3, .7]])
    out = viterbi_per_host(f, P, 0.5)
    assert out.tolist() == P.argmax(1).tolist()  # p_stay=0.5 -> no smoothing


def test_dbscan_1d():
    t = np.array([0, 1, 2, 3, 50, 100, 101, 102], float)
    lab = dbscan_1d(t, 1.5, 3)
    assert lab[0] == lab[3] and lab[4] == -1 and lab[5] == lab[7] != lab[0]


def test_cut_moves_into_gap():
    busy = np.array([[100, 200]])
    assert _move_cut_to_gap(150, busy, 100) in (99, 201)
    assert _move_cut_to_gap(150, busy, 10) == 150
