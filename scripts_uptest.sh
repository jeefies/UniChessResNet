#!/usr/bin/env bash
# 上行带宽测试：单流 vs 并行流。
# 下载方向用 8 连接提速了 3.8 倍，上行方向值得同样验证一次。
set -u
HOST=jeefy@172.16.2.12
D=/tmp/claude-1000/-home-jeefy-UniChess/9ebe8633-4d6f-4c96-9e20-cbf716d2d460/scratchpad/uptest
mkdir -p "$D"
SIZE=$((12 * 1024 * 1024))     # 每个测试文件 12MB

for i in 1 2 3 4 5 6; do
  [ -s "$D/f$i" ] || head -c $SIZE /dev/urandom > "$D/f$i"
done
ssh -o BatchMode=yes "$HOST" 'mkdir -p /tmp/uptest && rm -f /tmp/uptest/*' 2>/dev/null

mb() { echo "scale=2; $1 / $2" | bc; }

echo "=== 单流：12 MB ==="
t0=$(date +%s.%N)
scp -q -o BatchMode=yes -o Compression=no "$D/f1" "$HOST:/tmp/uptest/" 2>/dev/null
t1=$(date +%s.%N)
dt=$(echo "$t1 - $t0" | bc)
echo "  用时 ${dt}s  ->  $(mb 12 "$dt") MB/s"
SINGLE=$(mb 12 "$dt")

echo
echo "=== 4 并行流：4 x 12 MB = 48 MB ==="
ssh -o BatchMode=yes "$HOST" 'rm -f /tmp/uptest/*' 2>/dev/null
t0=$(date +%s.%N)
for i in 2 3 4 5; do
  scp -q -o BatchMode=yes -o Compression=no "$D/f$i" "$HOST:/tmp/uptest/f$i" 2>/dev/null &
done
wait
t1=$(date +%s.%N)
dt=$(echo "$t1 - $t0" | bc)
echo "  用时 ${dt}s  ->  $(mb 48 "$dt") MB/s 聚合"
PAR=$(mb 48 "$dt")

echo
echo "单流 ${SINGLE} MB/s  ->  4 并行 ${PAR} MB/s  (提速 $(echo "scale=2; $PAR / $SINGLE" | bc)x)"
ssh -o BatchMode=yes "$HOST" 'rm -rf /tmp/uptest' 2>/dev/null
