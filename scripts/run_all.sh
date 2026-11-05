#!/usr/bin/env bash
# Full pipeline E1-E8. Only continue past E3 if the gate says GO (see outputs/report.md).
#   bash scripts/run_all.sh            # everything
#   FORCE=1 bash scripts/run_all.sh    # ignore the gate
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_gate.sh "$@"
if ! grep -q "Decision:\*\* GO" outputs/report.md && [[ "${FORCE:-0}" != "1" ]]; then
  echo "Gate is not GO -> stopping after E3 (set FORCE=1 to continue anyway)."
  exit 0
fi
python scripts/run.py --exp E4 E5 E6 E7 E8 report "$@"
