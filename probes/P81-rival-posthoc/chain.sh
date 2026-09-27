#!/usr/bin/env bash
# Their France post-hoc chain (fr_posthoc -> fr_posthoc2 -> fr_posthoc3) on one of our bases.
# usage: chain.sh <base dir with matching_results.tsv + candidate_pairs.tsv> <out dir> [ph1 rules]
set -euo pipefail
S=/tmp/claude-1001/-home-shriya-github-repos-amc-26/c69ae552-b8a3-422d-89c9-9242b33640e4/scratchpad
T=$S/theirs/code/business_entity_resolution/scripts
W=/home/shriya/github_repos/amc-26/data/work/theirs
PY=$S/venv/bin/python
B=$1; O=$2; R1=${3:-+W,+b2,+d1,+d2g}
mkdir -p "$O"
export POLARS_MAX_THREADS=8
NORM=${NORM:-$W/normalized_v6}; RAW=${RAW:-$W/parquet}
N="--country ${CTRY:-France} --norm-dir $NORM --raw-dir $RAW"
$PY $T/fr_posthoc.py --matching "$B/matching_results.tsv" --candidates "$B/candidate_pairs.tsv" \
    --out-dir "$O" --tag ph1 --rules "$R1" --cap-scope country $N > "$O/ph1.log" 2>&1
$PY $T/fr_posthoc2.py --matching "$O/matching_results_ph1.tsv" --candidates "$O/candidate_pairs_ph1.tsv" \
    --out-dir "$O" --tag ph2 --rules W,F,E,S2,D --norm-dir-ph $NORM $N > "$O/ph2.log" 2>&1
$PY $T/fr_posthoc3.py --matching "$O/matching_results_ph2.tsv" --candidates "$O/candidate_pairs_ph2.tsv" \
    --out-dir "$O" --tag ph3 --model-output "$B/matching_results.tsv" --scores $S/noscores.parquet \
    --rules F1,F3,F4a,F4b,F4c,F5R,F5A,F2,F6,F7,F8 $N > "$O/ph3.log" 2>&1
echo "chain done: $O"
