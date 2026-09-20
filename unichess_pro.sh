#!/usr/bin/env bash
# PRO 6000 上的 UCI 引擎入口（供 cutechess / arena / calibrate 调用）。
#
# 环境变量：
#   UNICHESS_CKPT    权重文件（默认 Stage 1 最终版）
#   UNICHESS_DEVICE  cuda / cpu
#   UNICHESS_MCTS    >0 时启用 MCTS，值为每步模拟次数
#   UNICHESS_SYZYGY  残局表目录，设为空字符串可关闭（用于对照实验）
BASE=/root/autodl-tmp/fwj/UniChess
export PYTHONPATH="$BASE/pylibs${PYTHONPATH:+:$PYTHONPATH}"
cd "$BASE"
exec /root/autodl-tmp/conda_envs/hybridflow/bin/python uci.py \
  --ckpt "${UNICHESS_CKPT:-runs/stage1/ckpt_00187578.pt}" \
  --device "${UNICHESS_DEVICE:-cuda}" \
  ${UNICHESS_SYZYGY:+--syzygy "$UNICHESS_SYZYGY"} \
  ${UNICHESS_MCTS:+--mcts-sims $UNICHESS_MCTS} \
  "$@"
