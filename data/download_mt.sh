#!/usr/bin/env bash
# 多连接分块下载器。服务器支持 Range 请求，所以把文件切成固定大小的块并行取。
#
# 「定时重启」直接做进每块的 --max-time：超时就死，外层循环重新取，
# 不需要额外的监控进程。Lichess 对长连接似乎会限速，重连能拿到新配额。
#
# 断点续传：每块是独立的 .part 文件，已完整的块直接跳过，
# 未完成的块用 Range 从已有字节数接着下。全部齐了才拼装成最终文件。
#
# 用法: download_mt.sh <url> <输出文件> [并发数]
set -u
URL="${1:?用法: download_mt.sh <url> <输出文件> [并发数]}"
OUT="${2:?}"
JOBS="${3:-8}"
CHUNK=$((256 * 1024 * 1024))       # 256 MB/块
MAXTIME=600                        # 每块最多跑 10 分钟，到点重启
PARTDIR="${OUT}.parts"

TOTAL=$(curl -sIL --max-time 60 "$URL" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
[ -n "$TOTAL" ] || { echo "拿不到 content-length"; exit 1; }
NCHUNK=$(( (TOTAL + CHUNK - 1) / CHUNK ))
mkdir -p "$PARTDIR"
echo "总长 $TOTAL 字节，切 $NCHUNK 块 x 256MB，并发 $JOBS，每块超时 ${MAXTIME}s"

# 已有的连续前缀可以直接切块复用，避免重下
if [ -f "$OUT" ] && [ ! -f "$PARTDIR/.seeded" ]; then
  HAVE=$(stat -c%s "$OUT")
  FULL=$(( HAVE / CHUNK ))
  echo "发现已下前缀 $HAVE 字节 -> 可复用 $FULL 个完整块"
  for ((i=0; i<FULL; i++)); do
    if [ ! -f "$PARTDIR/p$i" ]; then
      dd if="$OUT" of="$PARTDIR/p$i" bs=1M skip=$((i*256)) count=256 status=none
    fi
  done
  touch "$PARTDIR/.seeded"
fi

fetch() {   # fetch <块号>
  local i=$1
  local start=$(( i * CHUNK ))
  local end=$(( start + CHUNK - 1 ))
  [ $end -ge $TOTAL ] && end=$(( TOTAL - 1 ))
  local want=$(( end - start + 1 ))
  local f="$PARTDIR/p$i"
  local have=0
  [ -f "$f" ] && have=$(stat -c%s "$f")
  [ "$have" -ge "$want" ] && return 0
  # --speed-limit/--speed-time: 低于 10KB/s 持续 60 秒就掐掉，换新连接
  curl -sL --max-time $MAXTIME --speed-limit 10240 --speed-time 60 \
       -r "$(( start + have ))-$end" "$URL" >> "$f" 2>/dev/null
  local now=0
  [ -f "$f" ] && now=$(stat -c%s "$f")
  [ "$now" -ge "$want" ]
}
export -f fetch
export URL CHUNK TOTAL PARTDIR MAXTIME

round=0
while :; do
  round=$((round+1))
  missing=()
  bytes=0
  for ((i=0; i<NCHUNK; i++)); do
    start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 ))
    [ $end -ge $TOTAL ] && end=$(( TOTAL - 1 ))
    want=$(( end - start + 1 )); have=0
    [ -f "$PARTDIR/p$i" ] && have=$(stat -c%s "$PARTDIR/p$i")
    bytes=$(( bytes + have ))
    [ "$have" -lt "$want" ] && missing+=("$i")
  done
  got=$(( NCHUNK - ${#missing[@]} ))
  echo "[$(date +%H:%M:%S)] 第 $round 轮: 完成 $got/$NCHUNK 块, 已有 $(( bytes / 1048576 )) MB / $(( TOTAL / 1048576 )) MB"
  [ ${#missing[@]} -eq 0 ] && break
  printf '%s\n' "${missing[@]}" | xargs -P "$JOBS" -I{} bash -c 'fetch {}'
done

echo "[$(date +%H:%M:%S)] 拼装 ..."
: > "$OUT.assembled"
for ((i=0; i<NCHUNK; i++)); do cat "$PARTDIR/p$i" >> "$OUT.assembled"; done
mv "$OUT.assembled" "$OUT"
FINAL=$(stat -c%s "$OUT")
if [ "$FINAL" = "$TOTAL" ]; then
  rm -rf "$PARTDIR"
  echo "[$(date +%H:%M:%S)] 完成: $OUT  $FINAL 字节（与远端一致）"
else
  echo "拼装后大小不符: $FINAL != $TOTAL"
  exit 1
fi
