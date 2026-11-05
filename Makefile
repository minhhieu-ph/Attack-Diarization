.PHONY: env check test smoke pilot data gate all report clean-smoke

CONFIG ?= configs/b300.yaml

env:            ## create conda env + CUDA-matched torch
	bash setup_env.sh

check:          ## verify packages and GPU
	python scripts/check_env.py

test:           ## unit tests (CPU, no torch needed)
	python -m pytest -q

smoke:          ## end-to-end on synthetic data
	bash scripts/smoke.sh

pilot:          ## feasibility pilot on CIDDS-001 (+ quick FlowDiar)
	python scripts/feasibility.py --config $(CONFIG) --flowdiar

data:           ## download + prepare all datasets
	python scripts/download.py --config $(CONFIG)
	python scripts/prepare.py --config $(CONFIG)

gate:           ## week 1: E1-E3 + gate decision
	bash scripts/run_gate.sh --config $(CONFIG)

all:            ## E1-E8 (stops after the gate unless GO)
	bash scripts/run_all.sh --config $(CONFIG)

report:
	python scripts/run.py --config $(CONFIG) --exp report

clean-smoke:
	rm -rf outputs_smoke dataset_smoke
