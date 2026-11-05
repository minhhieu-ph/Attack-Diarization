#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (a few minutes). Run this first on a new machine.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf outputs_smoke dataset_smoke
C="--config configs/smoke.yaml"
python scripts/make_synthetic.py $C
python scripts/prepare.py $C
python scripts/run.py $C --exp E1 E2 E3 E4 E5 E6 E7 E8 report
echo "Smoke test finished. See outputs_smoke/report.md"
