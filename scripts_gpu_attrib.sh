#!/usr/bin/env bash
# 测量「我的任务」对 GPU 的实际占用：
# 挂起自己的进程若干秒，对比挂起前后的 GPU 利用率，差值即为自己的贡献。
# nvidia-smi 不按进程拆分利用率，所以只能这样归因。
#
# 只挂起本项目的进程（calibrate.py / uci.py），绝不碰别人的任务。
set -u
SECS="${1:-12}"

mypids=$(pgrep -f "fwj/UniChess" 2>/dev/null | tr '\n' ' ')
mypids="$mypids $(pgrep -f 'eval/calibrate.py' 2>/dev/null | tr '\n' ' ')"
mypids="$mypids $(pgrep -f 'uci.py --ckpt' 2>/dev/null | tr '\n' ' ')"
mypids=$(echo "$mypids" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
echo "本项目进程: $mypids"

sample() {  # sample <标签> <次数>
  local label="$1" n="$2" sum=0 v
  for _ in $(seq 1 "$n"); do
    v=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    sum=$((sum + v))
    sleep 1
  done
  echo "$label 平均利用率: $((sum / n))%"
  echo "$((sum / n))"
}

echo "--- A. 双方都在跑 ---"
a=$(sample "  两者合计" "$SECS" | tail -1)

echo "--- B. 挂起本项目进程 ---"
for p in $mypids; do kill -STOP "$p" 2>/dev/null; done
sleep 2
b=$(sample "  仅对方任务" "$SECS" | tail -1)

echo "--- C. 恢复 ---"
for p in $mypids; do kill -CONT "$p" 2>/dev/null; done
echo "已恢复"

echo
echo "对方单独运行: ${b}%"
echo "两者合计:     ${a}%"
echo "本项目增量:   $((a - b)) 个百分点"
