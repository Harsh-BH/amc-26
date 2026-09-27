#!/usr/bin/env bash
# Official validator --check-ids on SageMaker (P38's solution_val.py, ml.m5.xlarge) for output/<sub>.
#   val59.sh <sub>      -> job name in data/work/p59/job_val_<sub>.txt
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
S=$1 TS=$(date +%m%d-%H%M) B=s3://sagemaker-114872709824-ap-south-1 P="--profile harsh --only-show-errors"
V=$B/amc26-p59/val/$S-$TS
aws s3 cp output/$S/matching_results.tsv $V/matching_$S.tsv $P
aws s3 cp output/$S/candidate_pairs.tsv $V/candidate_pairs.tsv $P
aws s3 cp data/student_resource/utils/validate_submission.py $V/validate_submission.py $P
.venv/bin/python - "$S" "$TS" "$V" <<'PY'
import sys; sys.path.insert(0, ".")
import sm_launch
s, ts, v = sys.argv[1:4]
sm_launch.quota_advice = lambda *a, **k: (1, None)       # ListServiceQuotas is rate-limited
sm_launch.main(["--profile", "harsh", "--solution", "probes/P38-qwen4b-fr/solution_val.py", "--no-spot", "--data-s3", v + "/",
                "--test-s3", "s3://sagemaker-114872709824-ap-south-1/amc26-p43/test/", "--instance", "ml.m5.xlarge",
                "--prefix", "amc26-p59", "--name", f"p59-val-{s.replace('_', '-')}-{ts}", "--max-run", "3600"])
open(f"data/work/p59/job_val_{s}.txt", "w").write(f"p59-val-{s.replace('_', '-')}-{ts}\n")
PY
