#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/local.yaml}
python run_bgfm.py brain-encoder --config "$CONFIG"
python run_bgfm.py gut-encoder --config "$CONFIG"
python run_bgfm.py brain-extract --config "$CONFIG"
python run_bgfm.py gut-extract --config "$CONFIG"
python run_bgfm.py align --config "$CONFIG"
python run_bgfm.py counterfactual --config "$CONFIG"
python run_bgfm.py classify --config "$CONFIG"
python run_bgfm.py subtype --config "$CONFIG"
