# dataset/

```
dataset/
├── raw/<name>/          # downloaded archives / CSV / parquet (scripts/download.py or by hand)
├── schedules/<name>.csv # attack schedule = episode ground truth for NetFlow v3 (you fill it)
├── interim/             # free scratch space
└── processed/<name>/    # written by scripts/prepare.py
    ├── flows.parquet    # one row per flow, canonical columns + f_* features + episode_id
    ├── episodes.parquet # one row per GT episode
    ├── splits.npz       # part index (0 train, 1 val, 2 test) per flow, per split
    └── meta.json        # counts, GT source, split info
```

## Datasets

| name | source | episode ground truth |
|---|---|---|
| `cidds001` | https://www.hs-coburg.de/wp-content/uploads/2024/11/CIDDS-001.zip (384 MB, direct link) | native `attackID` + attacker/victim role labels |
| `nf_unsw_nb15_v3`, `nf_bot_iot_v3`, `nf_ton_iot_v3`, `nf_cse_cic_ids2018_v3` | https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (UQ research data links) | `schedules/<name>.csv` |

The UQ links are landing pages. `scripts/download.py` tries them and looks for a
direct file link; if that fails it prints what to do: download the zip in a browser,
copy it to `raw/<name>/`, run `scripts/prepare.py`. A parquet mirror can be tried with
`python scripts/download.py --datasets <name> --source hf`.

## Why schedules matter

Episode labels must not be built with the same grouping rule that baseline B1 uses,
otherwise B1 matches the ground truth by construction. Order of preference
(`episodes.source_priority` in configs/default.yaml):

1. `attack_id` - dataset has a native attack id (CIDDS-001).
2. `schedule` - `schedules/<name>.csv` exists.
3. `heuristic` - fallback only. `meta.json` then says `"gt_independent": false` and the
   gate in `outputs/report.md` is reported as INVALID for that dataset.

## Schedule format

CSV, one row per attack execution (= one episode). `#` lines are comments.

| column | required | meaning |
|---|---|---|
| `start`, `end` | yes | epoch ms, or a date-time string (e.g. `2018-02-14 10:32:00`) |
| `attack` | yes | must equal the dataset's `Attack` label for those flows (case-insensitive) |
| `attackers` | no | `;`-separated attacker IPs; empty = any source |
| `victims` | no | `;`-separated victim IPs; empty = any destination |
| `tz` | no | time zone of date-time strings (first row is used), default UTC |

A flow is assigned to a row when it is labelled as attack with the same class, starts
within `[start - tol, end + tol]` (`episodes.schedule_tolerance_sec`) and matches the
listed hosts. `meta.json` reports flows matched by two rows and attack flows matched
by none. Start from `schedules/*.csv.template`, take times and IPs from each testbed's
own documentation, rename to `<name>.csv`.
