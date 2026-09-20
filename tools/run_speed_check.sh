#!/usr/bin/env bash
set -euo pipefail
cd /home/jeefy/UniChess
UNIT=unichess-train46-20260909.service
trap 'systemctl --user kill --kill-whom=main --signal=CONT "$UNIT"' EXIT
systemctl --user kill --kill-whom=main --signal=STOP "$UNIT"
/home/jeefy/miniconda3/envs/unichess/bin/python -u tools/bench_iteration.py
