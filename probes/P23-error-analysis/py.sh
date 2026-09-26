#!/usr/bin/env bash
# Memory-capped python for the P23 analysts: 1.2 GB (swap to 3 GB), OOM kills the command, never the session.
export BASH_ENV="$PWD/probes/oom_child.sh"
exec systemd-run --user --scope -q -p MemoryMax=1200M -p MemoryHigh=1080M -p MemorySwapMax=3G -p OOMPolicy=continue \
  "$PWD/.venv/bin/python" "$@"
