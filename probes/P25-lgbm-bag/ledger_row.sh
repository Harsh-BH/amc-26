#!/usr/bin/env bash
# ledger.log("exp", ...) without Python (= P22's ledger_row.sh): same row schema, same flock'd file.
# usage: ledger_row.sh NAME WORLD 'METRICS_JSON' 'CONFIG_JSON' HYPOTHESIS VERDICT RUNTIME_S PEAK_GB
set -euo pipefail
F=${AMC_LEDGER_DIR:-/home/harsh1/github-repos/amc-26/ledger}/experiments.jsonl
G=$(git -C "$(dirname "$0")" rev-parse --short HEAD)$(git -C "$(dirname "$0")" diff --quiet HEAD -- 2>/dev/null || echo -dirty)
L=$(jq -nc --arg ts "$(date +%Y-%m-%dT%H:%M:%S%z)" --arg git "$G" --arg name "$1" --arg world "$2" --argjson m "$3" \
      --argjson c "$4" --arg h "$5" --arg v "$6" --argjson rt "$7" --argjson pk "$8" \
      '{ts: $ts, git: $git, probe: "P25-lgbm-bag", subproblem: "M2 LightGBM bagging/boosting ensemble", name: $name,
        world: $world, metrics: $m, config: $c, hypothesis: $h, verdict: $v, runtime_s: $rt, peak_gb: $pk}')
{ flock 9; printf "%s\n" "$L" >&9; } 9>>"$F"
echo "$L" | cut -c1-200
