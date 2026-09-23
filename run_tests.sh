#!/usr/bin/env bash
# Phase A 全套验收
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "=========== UniChess Phase A 验收 ==========="
echo; echo "--- A2 局面/走法编解码往返 ---"; $PY unichess_r/core/test_roundtrip.py
echo; echo "--- A4 96 字节记录往返 ---";      $PY data/test_record.py
echo; echo "--- A5 训练快路径 vs 参考实现 ---"; $PY unichess_r/model/test_dataset.py
echo; echo "--- A6 残局表收官 ---";            $PY unichess_r/engine/test_tablebase.py
if [ -d data/shards_evals ]; then echo; echo "--- evals 分片正确性 ---"; $PY data/test_build_evals.py data/shards_evals; fi
echo; echo "=========== 全部通过 ==========="
