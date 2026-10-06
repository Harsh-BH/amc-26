#!/usr/bin/env bash
# Official validator --check-ids on SageMaker (P38's solution_val.py, ml.m5.xlarge) for output/<sub>... sharing one
# candidate_pairs.tsv (taken from the first).   val79.sh sub79q sub79   -> data/work/p79/job_val.txt (P79 copy of P72's val72.sh)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
TS=$(date +%m%d-%H%M) B=s3://sagemaker-114872709824-ap-south-1 P="--profile harsh --only-show-errors"
V=$B/amc26-p79/val/$TS
for S in "$@"; do aws s3 cp output/$S/matching_results.tsv $V/matching_$S.tsv $P; done
aws s3 cp output/$1/candidate_pairs.tsv $V/candidate_pairs.tsv $P
aws s3 cp data/student_resource/utils/validate_submission.py $V/validate_submission.py $P
.venv/bin/python - "$TS" "$V" <<'PY'
import sys; sys.path.insert(0, ".")
import sm_launch
ts, v = sys.argv[1:3]
sm_launch.quota_advice = lambda *a, **k: (1, None)       # ListServiceQuotas is rate-limited
sm_launch.main(["--profile", "harsh", "--solution", "probes/P38-qwen4b-fr/solution_val.py", "--no-spot", "--data-s3", v + "/",
                "--test-s3", "s3://sagemaker-114872709824-ap-south-1/amc26-p43/test/", "--instance", "ml.m5.xlarge",
                "--prefix", "amc26-p79", "--name", f"p79-val-{ts}", "--max-run", "3600"])
open("data/work/p79/job_val.txt", "w").write(f"p79-val-{ts}\n")
PY
