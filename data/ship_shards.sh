#!/usr/bin/env bash
# 把训练分片压缩后送到 GPU 机。
#
# 为什么要压缩：本机 -> GPU 机是一条 0.67 MB/s、RTT 75ms 的 WAN 链路
# （不是局域网，实测 ping 75ms）。12.29 GB 直传要 5 小时；
# zstd -9 有 3.19x 压缩率，压完约 3.85 GB，降到约 1.6 小时。
#
# 按分片编号顺序传，GPU 机上凑够若干个就能先开训，不必等全部到齐。
# 可重复执行：已传过且大小一致的分片会跳过。
set -u
SRC="${1:-data/shards_evals}"
HOST="${2:-jeefy@172.16.2.12}"
DST="${3:-~/UniChess/data/shards_evals}"
LVL="${4:-9}"
# 远端解压用的 python。5070Ti 用 conda unichess；
# PRO 6000 上共享环境不能装包，走 PYTHONPATH 指向项目自己的 pylibs。
RPY="${5:-~/miniconda3/envs/unichess/bin/python}"
RPYPATH="${6:-}"

CACHE="/tmp/claude-1000/-home-jeefy-UniChess/9ebe8633-4d6f-4c96-9e20-cbf716d2d460/scratchpad/zst"
mkdir -p "$CACHE"
ssh -o BatchMode=yes "$HOST" "mkdir -p $DST"
# 把 DST 解析成远端绝对路径。~ 在远端 shell 里会展开，但塞进 python -c 的
# 字符串字面量后不会——之前所有解压都因此 FileNotFoundError。
DST=$(ssh -o BatchMode=yes "$HOST" "cd $DST && pwd")
echo "远端目录解析为: $DST"

PY=.venv/bin/python
total=$(ls "$SRC"/*.bin 2>/dev/null | wc -l)
i=0
for f in $(ls "$SRC"/*.bin | sort); do
  i=$((i+1))
  base=$(basename "$f" .bin)
  zf="$CACHE/$base.bin.zst"

  # 1) 压缩（已压过就复用）
  if [ ! -s "$zf" ]; then
    $PY -c "
import zstandard as zstd, sys
data = open('$f','rb').read()
open('$zf','wb').write(zstd.ZstdCompressor(level=$LVL, threads=8).compress(data))
"
  fi

  # 2) 若远端已有同名 .bin 且大小一致就跳过
  want=$(stat -c%s "$f")
  have=$(ssh -o BatchMode=yes "$HOST" "stat -c%s $DST/$base.bin 2>/dev/null || echo 0")
  if [ "$have" = "$want" ]; then
    echo "[$i/$total] $base 已在远端，跳过"
    continue
  fi

  # 3) 传输 + 远端解压（rsync 断点续传；不用 -z，内容已压过）
  echo "[$(date +%H:%M:%S)] [$i/$total] 传 $base ($(stat -c%s "$zf" | awk '{printf "%.0f MB", $1/1048576}')) ..."
  rsync -a --partial --inplace --timeout=300 "$zf" "$HOST:$DST/" || { echo "  传输失败，稍后重试"; continue; }
  ssh -o BatchMode=yes "$HOST" "
    ${RPYPATH:+PYTHONPATH=$RPYPATH} $RPY -c \"
import zstandard as zstd
d = zstd.ZstdDecompressor().decompress(open('$DST/$base.bin.zst','rb').read(), max_output_size=1<<31)
open('$DST/$base.bin','wb').write(d)
\" && rm -f $DST/$base.bin.zst"
  got=$(ssh -o BatchMode=yes "$HOST" "stat -c%s $DST/$base.bin 2>/dev/null || echo 0")
  if [ "$got" = "$want" ]; then
    echo "  ✓ $base 落地 $((got/1048576)) MB"
    rm -f "$zf"
  else
    echo "  ✗ $base 大小不符 ($got != $want)"
  fi
done
echo "[$(date +%H:%M:%S)] 全部完成"
ssh -o BatchMode=yes "$HOST" "ls $DST/*.bin 2>/dev/null | wc -l | xargs echo 远端分片数:; du -sh $DST"
