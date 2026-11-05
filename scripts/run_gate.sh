#!/usr/bin/env bash
# Week 1: download -> prepare -> E1 (GT stats) -> E2 (baselines) -> E3 (errors) -> gate decision.
# Extra args are passed to every python call, e.g.:
#   bash scripts/run_gate.sh --datasets cidds001 nf_ton_iot_v3
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/download.py "$@" || true
python scripts/prepare.py "$@"
python scripts/run.py --exp E1 E2 E3 report "$@"
echo "Gate decision is at the top of outputs/report.md"
