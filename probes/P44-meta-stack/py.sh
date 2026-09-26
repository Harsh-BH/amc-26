#!/usr/bin/env bash
# Memory-capped python for P44-meta-stack: 2.2 GB (swap to 3 GB); an OOM kills the command, never the session.
export BASH_ENV="$PWD/probes/oom_child.sh"
exec systemd-run --user --scope -q -p MemoryMax=2200M -p MemoryHigh=2000M -p MemorySwapMax=3G -p OOMPolicy=continue \
  "$PWD/.venv/bin/python" "$@"
