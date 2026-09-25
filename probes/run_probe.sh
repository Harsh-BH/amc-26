#!/usr/bin/env bash
# One headless Opus 5.5 probe: memory-capped, rendered live in the tmux pane, DONE marker on exit.
# MODE=new starts the session from _COMMON.md + BRIEF.md; MODE=resume continues it with RESUME.md.
# The body is one function called on the last line: bash parses it whole before running, so replacing this
# file while probes run cannot derail them (P3 lost its DONE marker to an in-place edit at 04:40).
main() {
  set -uo pipefail
  local ID=$1 MEM=$2 MODE=${3:-new}; local DIR=probes/$ID; local SID; SID=$(cat "$DIR/session_id")
  # Tool commands run in bash, which sources BASH_ENV: they raise their own oom_score_adj to 1000, so the
  # kernel OOM killer picks them, not the claude CLI (it killed the CLI in P2 at 04:13 when it was the biggest).
  export SHELL=/usr/bin/bash BASH_ENV="$PWD/probes/oom_child.sh"
  export CLAUDE_CODE_SUBAGENT_MODEL=claude-opus-5-5
  # worktree probes log to the MAIN checkout's ledger (one ledger, no merge conflicts on it)
  local MAIN; MAIN=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
  export AMC_LEDGER_DIR="$MAIN/ledger"
  # swap: MemoryHigh (90% of the cap) makes the kernel reclaim/swap before the hard cap, and up to 6G of swap
  # lets a probe spill instead of being OOM-killed (13.7G swap free; verified: 500MB in a 200M cap survives)
  local HIGH=$(( $(numfmt --from=iec "$MEM") * 9 / 10 ))
  local prompt sess
  if [[ $MODE == resume ]]; then prompt=("$DIR/RESUME.md"); sess=(--resume "$SID")
  else prompt=(probes/_COMMON.md "$DIR/BRIEF.md"); sess=(--session-id "$SID"); fi
  # OOMPolicy=continue: an oversized command is OOM-killed (exit 137) but the session survives and adapts;
  # systemd's default (stop) tears down the whole scope, i.e. the probe itself (exit 143).
  cat "${prompt[@]}" \
  | systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemoryHigh="$HIGH" -p MemorySwapMax=6G -p OOMPolicy=continue \
      claude -p --model claude-opus-5-5 --effort xhigh --dangerously-skip-permissions \
             "${sess[@]}" -n "$ID" --output-format stream-json --verbose \
  | tee -a "$DIR/stream.jsonl" \
  | jq -rj --unbuffered '
      if .type=="assistant" then (.message.content[] |
          if .type=="text" then .text+"\n"
          elif .type=="tool_use" then "  -> \(.name) \((.input.command // .input.file_path // "")|tostring|.[0:120])\n"
          else empty end)
      elif .type=="result" then "\n== \(.subtype)  cost $\(.total_cost_usd)\n" else empty end'
  local rc=${PIPESTATUS[1]}
  # exit codes lie when the scope is torn down; only a final result event means the session really finished
  local last; last=$(tail -n1 "$DIR/stream.jsonl" | jq -r 'if .type=="result" then "result:\(.subtype)" else "CRASHED(last=\(.type))" end' 2>/dev/null)
  echo "exit=$rc $last $(date -Is)" > "$DIR/DONE"
}
main "$@"; exit
