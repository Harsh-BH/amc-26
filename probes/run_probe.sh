#!/usr/bin/env bash
# One headless Opus 5.5 probe: memory-capped, rendered live in the tmux pane, DONE marker with exit code.
set -uo pipefail
ID=$1; MEM=$2; DIR=probes/$ID; SID=$(cat "$DIR/session_id")
cat probes/_COMMON.md "$DIR/BRIEF.md" \
| systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 \
    claude -p --model claude-opus-5-5 --effort xhigh --dangerously-skip-permissions \
           --session-id "$SID" -n "$ID" --output-format stream-json --verbose \
| tee -a "$DIR/stream.jsonl" \
| jq -rj --unbuffered '
    if .type=="assistant" then (.message.content[] |
        if .type=="text" then .text+"\n"
        elif .type=="tool_use" then "  -> \(.name) \((.input.command // .input.file_path // "")|tostring|.[0:120])\n"
        else empty end)
    elif .type=="result" then "\n== \(.subtype)  cost $\(.total_cost_usd)\n" else empty end'
echo "exit=${PIPESTATUS[1]} $(date -Is)" > "$DIR/DONE"
