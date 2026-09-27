#!/usr/bin/env bash
# Memory-capped python for P40: 2.2 GB (swap to 3 GB); an OOM kills the command, never the session.
export BASH_ENV="$PWD/probes/oom_child.sh"
exec systemd-run --user --scope -q -p MemoryMax=2600M -p MemoryHigh=2400M -p MemorySwapMax=3G -p OOMPolicy=continue \
  "$PWD/.venv/bin/python" "$@"
