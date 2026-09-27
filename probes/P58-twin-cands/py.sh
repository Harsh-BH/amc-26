#!/usr/bin/env bash
# Memory-capped python for P58: 1.3 GB (swap to 2 GB); an OOM kills the command, never the session.
export BASH_ENV="$PWD/probes/oom_child.sh"
exec systemd-run --user --scope -q -p MemoryMax=1300M -p MemoryHigh=1250M -p MemorySwapMax=2G -p OOMPolicy=continue \
  "$PWD/.venv/bin/python" "$@"
