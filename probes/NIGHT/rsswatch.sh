#!/bin/bash
# Kill any .venv python whose RSS exceeds LIMIT MB (probe jobs), so agy agents themselves are never OOM-killed.
LIMIT=${1:-2200}; LOG=/tmp/claude-1000/rsswatch.log
while true; do
  for p in $(pgrep -f "\.venv/bin/python"); do
    r=$(awk '/VmRSS/{print int($2/1024)}' /proc/$p/status 2>/dev/null)
    if [ -n "$r" ] && [ "$r" -gt "$LIMIT" ]; then
      echo "$(date +%T) kill $p rss=${r}M $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-120)" >> $LOG; kill -9 $p
    fi
  done
  sleep 2
done
