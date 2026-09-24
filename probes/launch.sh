#!/usr/bin/env bash
# usage: probes/launch.sh <probe-id> [MemoryMax, default 3G] [new|resume]
#   new    — needs probes/<id>/BRIEF.md, starts a fresh session
#   resume — needs probes/<id>/RESUME.md, continues the probe's existing session (context intact)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
ID=$1; MEM=${2:-3G}; MODE=${3:-new}; DIR=probes/$ID
if [[ $MODE == resume ]]; then
  [[ -f $DIR/RESUME.md && -f $DIR/session_id ]] || { echo "resume needs $DIR/RESUME.md and a session_id" >&2; exit 1; }
else
  [[ -f $DIR/BRIEF.md ]] || { echo "missing $DIR/BRIEF.md" >&2; exit 1; }
fi
if [[ -e $DIR/session_id && ! -e $DIR/DONE ]]; then echo "$ID is still running" >&2; exit 1; fi
need=$(( $(numfmt --from=iec "$MEM") + 1024**3 + 700*1024**2 ))     # cap + 1G headroom + CLI
avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) * 1024 ))
(( avail > need )) || { echo "RAM: $((avail>>20)) MiB available, $ID needs $((need>>20)) MiB" >&2; exit 1; }
rm -f "$DIR/DONE" "$DIR/.collected"
if [[ $MODE == resume ]]; then touch "$DIR/session_id"; else uuidgen > "$DIR/session_id"; fi   # mtime = launch time (watch.sh grace)
tmux has-session -t amc 2>/dev/null || tmux new-session -d -s amc -n master
tmux kill-window -t "amc:$ID" 2>/dev/null || true
tmux new-window -d -t amc -n "$ID" -c "$PWD" "probes/run_probe.sh $ID $MEM $MODE; exec bash"
echo "launched $ID ($MODE)  session=$(cat "$DIR/session_id")  watch: tmux attach -t amc"
