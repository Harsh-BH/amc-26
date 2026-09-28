#!/usr/bin/env bash
# One command: raw TSVs -> the final submission (sub90u, public LB 0.989382), then every check. Needs the trained
# artefacts of README §2 (run.py retrains 4 of them from the training data; the others come from the GPU /
# experiment jobs listed there).
#   bash code/business_entity_resolution/src/reproduce.sh [PYTHON]        (run from the repository root)
# Needs the setup of README §1 (data under data/student_resource/, trained artefacts or time to rebuild them).
set -euo pipefail
PY=${1:-.venv/bin/python}
SRC=code/business_entity_resolution/src
"$PY" "$SRC/run.py" --split test --stage model --recipe sub66_final --drop-sibling-cands --out output/sub66_final_small
"$PY" "$SRC/final_build.py" output/sub66_final_small output/sub77 --upto 41      # recorded France change sets
"$PY" "$SRC/final_patch.py" output/sub77 output/sub90u                           # recorded evening change sets
"$PY" "$SRC/verify_outputs.py" output/sub90u                                     # invariants + headline numbers
echo "a8ad38b59e75013bcb53ec13578b5529  output/sub90u/matching_results.tsv
f311dc4d44e4587ece2e74541072c2e2  output/sub90u/candidate_pairs.tsv" | md5sum -c -   # = the leaderboard upload
(cd data/student_resource && python3 utils/validate_submission.py --matching ../../output/sub90u/matching_results.tsv \
    --candidate ../../output/sub90u/candidate_pairs.tsv --test-dir dataset/test --check-ids)
