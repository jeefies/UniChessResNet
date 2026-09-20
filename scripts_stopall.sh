#!/usr/bin/env bash
# 停掉所有后台传输任务。
# 注意：不能在调用它的命令行里出现 ship/download 这些字样，
# 否则 pgrep -f 会匹配到调用者自己的 shell 并把它杀掉（之前踩过两次）。
MYPID=$$
PPID_=$(ps -o ppid= -p $MYPID | tr -d ' ')

kill_pat() {
  for p in $(pgrep -f "$1" 2>/dev/null); do
    [ "$p" = "$MYPID" ] && continue
    [ "$p" = "$PPID_" ] && continue
    kill "$p" 2>/dev/null
  done
}
kill_pat 'ship_shards'
kill_pat 'download_mt'
for p in $(pgrep -x rsync 2>/dev/null; pgrep -x curl 2>/dev/null; pgrep -x scp 2>/dev/null); do
  kill "$p" 2>/dev/null
done
sleep 2
echo "剩余 rsync=$(pgrep -x rsync 2>/dev/null | wc -l) curl=$(pgrep -x curl 2>/dev/null | wc -l) scp=$(pgrep -x scp 2>/dev/null | wc -l)"
