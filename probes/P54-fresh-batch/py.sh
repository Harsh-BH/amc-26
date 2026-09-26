#!/usr/bin/env bash
# Memory-capped python for P54: 1.9 GB (swap to 2 GB); an OOM kills the command, never the session.
export BASH_ENV="$PWD/probes/oom_child.sh"
exec systemd-run --user --scope -q -p MemoryMax=1900M -p MemoryHigh=1800M -p MemorySwapMax=2G -p OOMPolicy=continue \
  "$PWD/.venv/bin/python" "$@"
