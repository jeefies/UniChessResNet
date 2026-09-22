#!/usr/bin/env bash
# A1: Syzygy 3-4-5 子残局表（约 0.94GB，145 组 WDL+DTZ）
set -u
BASE="http://tablebase.sesse.net/syzygy/3-4-5/"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/syzygy345"
mkdir -p "$DEST"

curl -sL --max-time 60 "$BASE" \
  | grep -oE 'href="[^"]+\.rtb[wz]"' | sed 's/href="//;s/"$//' | sort -u > /tmp/syzygy_files.txt
total=$(wc -l < /tmp/syzygy_files.txt)
echo "共 $total 个文件"

i=0
while read -r f; do
  i=$((i+1))
  if [ -s "$DEST/$f" ]; then continue; fi
  curl -sL -C - --retry 5 --retry-all-errors -o "$DEST/$f" "$BASE$f"
  [ $((i % 25)) -eq 0 ] && echo "[$(date +%H:%M:%S)] $i/$total"
done < /tmp/syzygy_files.txt

echo "[$(date +%H:%M:%S)] SYZYGY DONE: $(ls "$DEST" | wc -l) 文件, $(du -sh "$DEST" | cut -f1)"
