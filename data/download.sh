#!/usr/bin/env bash
# A1: 原始数据下载。全部可断点续传（curl -C -），重复执行安全。
set -u
RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$RAW"

get() {  # get <url> <dest>
  local url="$1" dest="$2"
  echo "[$(date +%H:%M:%S)] --> $dest"
  curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
       -o "$RAW/$dest" "$url" 2>&1 | tail -1
  echo "[$(date +%H:%M:%S)] <-- $dest  $(du -h "$RAW/$dest" 2>/dev/null | cut -f1)"
}

get "https://database.lichess.org/lichess_db_puzzle.csv.zst"                       lichess_db_puzzle.csv.zst
get "https://database.lichess.org/lichess_db_eval.jsonl.zst"                       lichess_db_eval.jsonl.zst
get "https://database.lichess.org/standard/lichess_db_standard_rated_2026-07.pgn.zst" lichess_db_standard_rated_2026-07.pgn.zst
echo "[$(date +%H:%M:%S)] ALL DONE"
ls -lh "$RAW"
