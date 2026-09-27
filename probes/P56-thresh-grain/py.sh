#!/usr/bin/env bash
# Memory-capped python for P56: 1.2 GB (swap to 1.5 GB); an OOM kills the command, never the session.
export BASH_ENV="$PWD/probes/oom_child.sh"
exec systemd-run --user --scope -q -p MemoryMax=1200M -p MemoryHigh=1100M -p MemorySwapMax=1500M -p OOMPolicy=continue \
  "$PWD/.venv/bin/python" "$@"
