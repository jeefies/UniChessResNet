#!/usr/bin/env bash
# 5070 Ti 上的 UCI 引擎入口（cutechess / arena / calibrate 调用它）。
#
# 环境变量：
#   UNICHESS_CKPT    权重（默认 Stage 1 最终版）
#   UNICHESS_DEVICE  cuda / cpu
#   UNICHESS_MCTS    >0 启用 MCTS，值为每步模拟次数；不设或为 0 则策略网络直出
#   UNICHESS_SYZYGY  残局表目录；**设为空字符串即关闭**（用于做对照实验）
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unichess

args=(--ckpt "${UNICHESS_CKPT:-runs/stage1/ckpt_00187578.pt}"
      --device "${UNICHESS_DEVICE:-cuda}")
# 用 ${VAR+set} 判断「是否显式设过」，这样空字符串能表达「明确关闭」，
# 而不是被 :- 默认值悄悄替换回去
if [ -n "${UNICHESS_SYZYGY+set}" ]; then
  [ -n "$UNICHESS_SYZYGY" ] && args+=(--syzygy "$UNICHESS_SYZYGY")
else
  args+=(--syzygy data/raw/syzygy345)
fi
[ "${UNICHESS_MCTS:-0}" -gt 0 ] 2>/dev/null && args+=(--mcts-sims "$UNICHESS_MCTS")

exec python uci.py "${args[@]}" "$@"
