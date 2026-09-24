#!/usr/bin/env bash
# usage: probes/launch.sh <probe-id> [MemoryMax, default 3G]   — needs probes/<id>/BRIEF.md
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
ID=$1; MEM=${2:-3G}; DIR=probes/$ID
[[ -f $DIR/BRIEF.md ]] || { echo "missing $DIR/BRIEF.md" >&2; exit 1; }
if [[ -e $DIR/session_id && ! -e $DIR/DONE ]]; then echo "$ID is still running" >&2; exit 1; fi
need=$(( $(numfmt --from=iec "$MEM") + 1024**3 + 700*1024**2 ))     # cap + 1G headroom + CLI
avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) * 1024 ))
(( avail > need )) || { echo "RAM: $((avail>>20)) MiB available, $ID needs $((need>>20)) MiB" >&2; exit 1; }
rm -f "$DIR/DONE" "$DIR/.collected"; uuidgen > "$DIR/session_id"
tmux has-session -t amc 2>/dev/null || tmux new-session -d -s amc -n master
tmux new-window -d -t amc -n "$ID" -c "$PWD" "probes/run_probe.sh $ID $MEM; exec bash"
echo "launched $ID  session=$(cat "$DIR/session_id")  watch: tmux attach -t amc"
