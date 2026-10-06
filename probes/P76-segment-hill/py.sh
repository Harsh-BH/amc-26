#!/usr/bin/env bash
# Memory-capped python for P76: 2.4 GB (swap to 1.5 GB); an OOM kills the command, never the session.
export BASH_ENV="$PWD/probes/oom_child.sh"
exec systemd-run --user --scope -q -p MemoryMax=2400M -p MemoryHigh=2200M -p MemorySwapMax=2600M -p OOMPolicy=continue \
  "$PWD/.venv/bin/python" "$@"
