#!/usr/bin/env bash
# Variant A: their ADD rules only (no drops; no rule c, no F6), F7 number guard on their adds.
# usage: chainA.sh <base dir> <out dir>   env: CTRY NORM RAW
set -euo pipefail
S=/tmp/claude-1001/-home-shriya-github-repos-amc-26/c69ae552-b8a3-422d-89c9-9242b33640e4/scratchpad
T=$S/theirs/code/business_entity_resolution/scripts; W=/home/shriya/github_repos/amc-26/data/work/theirs; PY=$S/venv/bin/python
B=$1; O=$2; mkdir -p "$O"; export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-4}
NORM=${NORM:-$W/normalized_v6}; RAW=${RAW:-$W/parquet}
N="--country ${CTRY:-France} --norm-dir $NORM --raw-dir $RAW"
$PY $T/fr_posthoc.py --matching "$B/matching_results.tsv" --candidates "$B/candidate_pairs.tsv" --out-dir "$O" --tag ph1 \
    --rules a,b,b2,d1,d2g,cap --cap-scope country $N > "$O/ph1.log" 2>&1
$PY $T/fr_posthoc2.py --matching "$O/matching_results_ph1.tsv" --candidates "$O/candidate_pairs_ph1.tsv" --out-dir "$O" --tag ph2 \
    --rules F,E,D --norm-dir-ph $NORM $N > "$O/ph2.log" 2>&1
$PY $T/fr_posthoc3.py --matching "$O/matching_results_ph2.tsv" --candidates "$O/candidate_pairs_ph2.tsv" --out-dir "$O" --tag ph3 \
    --model-output "$B/matching_results.tsv" --scores $S/noscores.parquet --rules F5A,F2,F7,F8 $N > "$O/ph3.log" 2>&1
echo "chainA done: $O"
