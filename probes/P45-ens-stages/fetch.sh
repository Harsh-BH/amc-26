#!/usr/bin/env bash
# fetch.sh v1 v2 ...: holdout preds of finished P45 matcher jobs -> data/work/p45/pred/<v>_holdout_<c>.parquet
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
for v in "$@"; do
  j=$(cat data/work/p45/job_$v.txt)
  for c in India US; do
    aws --profile harsh s3 cp --only-show-errors s3://sagemaker-114872709824-ap-south-1/amc26-p45/results/$j/pred/holdout_$c.parquet data/work/p45/pred/${v}_holdout_$c.parquet
  done
  echo "$v: $(aws --profile harsh s3 cp s3://sagemaker-114872709824-ap-south-1/amc26-p45/results/$j/status.json - | jq -c '{trees, best_iter, best_val_loss, best_step, elapsed_s}')"
done
