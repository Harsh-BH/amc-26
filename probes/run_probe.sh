#!/usr/bin/env bash
# One headless Opus 5.5 probe: memory-capped, rendered live in the tmux pane, DONE marker with exit code.
# MODE=new starts the session from _COMMON.md + BRIEF.md; MODE=resume continues it with RESUME.md.
# OOMPolicy=continue: an oversized command is OOM-killed (exit 137) but the session survives and adapts;
# systemd's default (stop) tears down the whole scope, i.e. the probe itself (exit 143).
set -uo pipefail
ID=$1; MEM=$2; MODE=${3:-new}; DIR=probes/$ID; SID=$(cat "$DIR/session_id")
if [[ $MODE == resume ]]; then prompt=("$DIR/RESUME.md"); sess=(--resume "$SID")
else prompt=(probes/_COMMON.md "$DIR/BRIEF.md"); sess=(--session-id "$SID"); fi
cat "${prompt[@]}" \
| systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 -p OOMPolicy=continue \
    claude -p --model claude-opus-5-5 --effort xhigh --dangerously-skip-permissions \
           "${sess[@]}" -n "$ID" --output-format stream-json --verbose \
| tee -a "$DIR/stream.jsonl" \
| jq -rj --unbuffered '
    if .type=="assistant" then (.message.content[] |
        if .type=="text" then .text+"\n"
        elif .type=="tool_use" then "  -> \(.name) \((.input.command // .input.file_path // "")|tostring|.[0:120])\n"
        else empty end)
    elif .type=="result" then "\n== \(.subtype)  cost $\(.total_cost_usd)\n" else empty end'
echo "exit=${PIPESTATUS[1]} $(date -Is)" > "$DIR/DONE"
