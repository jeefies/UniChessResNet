#!/usr/bin/env bash
set -euo pipefail
cd /home/jeefy/UniChess
PY=/home/jeefy/miniconda3/envs/unichess/bin/python
OUT=runs/iteration46_20260909
UNIT=unichess-train46-fast-20260909
if systemctl --user is-active --quiet "$UNIT.service"; then
  echo 'Training already active'; exit 0
fi
mkdir -p "$OUT"
for name in train_pools.npz train_pools.json valid_pools.npz valid_pools.json; do
  if [ ! -f "$OUT/$name" ]; then cp "runs/iteration46_smoke_20260909/$name" "$OUT/$name"; fi
done
resume=()
if [ -f "$OUT/latest.pt" ]; then resume=(--resume); fi
systemd-run --user --unit="$UNIT" --description='UniChess 46.34M supervised iteration' \
  --property=WorkingDirectory=/home/jeefy/UniChess \
  --property=Nice=10 \
  --property=TimeoutStopSec=180 \
  --property=StandardOutput=append:/home/jeefy/UniChess/$OUT/train-fast.log \
  --property=StandardError=append:/home/jeefy/UniChess/$OUT/train-fast.log \
  "$PY" -u model/train_iteration_fast.py --out "$OUT" \
  --steps 125000 --batch 512 --accum 2 --save-every 1000 "${resume[@]}"
systemctl --user show "$UNIT" -p ActiveState -p MainPID -p ExecMainStatus
