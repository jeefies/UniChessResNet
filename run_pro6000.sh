#!/usr/bin/env bash
# PRO 6000 (AutoDL) 上的运行入口。
#
# 三条约束，都来自机器是共享的：
#   1. 只写 ~/autodl-tmp/fwj/UniChess，不碰任何其他目录
#   2. 复用共享 conda 环境但**不往里装包**——依赖装在项目自己的 pylibs 下，
#      靠 PYTHONPATH 引入，对环境零侵入
#   3. 卡上有别人的任务（实测占 25.6 GB / 91% 利用率），
#      所以用小 batch、不独占显存，并且不要杀任何进程
set -u
BASE=/root/autodl-tmp/fwj/UniChess
export PYTHONPATH="$BASE/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/autodl-tmp/conda_envs/hybridflow/bin/python
cd "$BASE"
exec "$PY" "$@"
