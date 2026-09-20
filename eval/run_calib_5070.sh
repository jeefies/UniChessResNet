#!/usr/bin/env bash
# 在 5070 Ti 上跑 Elo 标定。
#
# 用法:
#   run_calib_5070.sh <标签> <MCTS模拟数> <档位列表> [残局表目录或空]
# 例:
#   run_calib_5070.sh mcts1600_notb 1600 1900,2200,2500 ""
#   run_calib_5070.sh mcts1600_tb   1600 1900,2200,2500 data/raw/syzygy345
set -u
TAG="${1:?需要标签}"
SIMS="${2:-1600}"
ELOS="${3:-1900,2200,2500}"
SYZ="${4-}"

cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unichess
mkdir -p logs runs

export UNICHESS_MCTS="$SIMS"
export UNICHESS_SYZYGY="$SYZ"
export UNICHESS_CKPT="${UNICHESS_CKPT:-runs/stage1/ckpt_00187578.pt}"

echo "标签=$TAG  模拟数=$SIMS  档位=$ELOS  残局表=${SYZ:-关闭}"
nohup python eval/calibrate.py \
  --engine ./unichess_gpu.sh \
  --stockfish tools/stockfish \
  --elos "$ELOS" --pairs 30 --movetime 0.5 --workers 4 \
  --openings eval/openings.txt \
  --out "runs/calib_$TAG.json" \
  > "logs/calib_$TAG.log" 2>&1 &
echo "已启动 pid=$!  日志 logs/calib_$TAG.log"
