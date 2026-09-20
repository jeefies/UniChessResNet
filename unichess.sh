#!/usr/bin/env bash
# cutechess-cli / Arena 的引擎入口
cd "$(dirname "$0")"
exec .venv/bin/python uci.py \
  ${UNICHESS_MCTS:+--mcts-sims $UNICHESS_MCTS} \
  --ckpt "${UNICHESS_CKPT:-runs/smoke/ckpt_00000200.pt}" \
  --device "${UNICHESS_DEVICE:-cpu}" \
  --syzygy "${UNICHESS_SYZYGY:-data/raw/syzygy345}" \
  "$@"
