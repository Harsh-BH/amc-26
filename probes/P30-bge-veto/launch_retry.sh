#!/bin/bash
# Retry the P30 launch every 60 s until the single ml.g5.xlarge training slot is free (other probes share it).
cd "$(dirname "$0")/../.."
DS=s3://sagemaker-114872709824-ap-south-1/amc26-p30/data/p30-veto-0926-1647/train/
for i in $(seq 1 120); do
  J=p30-veto-$(date +%m%d-%H%M)
  if .venv/bin/python probes/P30-bge-veto/p30.py launch --profile harsh --region ap-south-1 --instance ml.g5.xlarge \
      --no-spot --prefix amc26-p30 --name $J --data-s3 $DS --solution probes/P30-bge-veto/solution_p30.py --max-run 7200 \
      --hp "{\"phases\": [\"loco_US\", \"loco_India\", \"ckpt\"], \"ckpt\": \"s3://sagemaker-114872709824-ap-south-1/amc26-p13/results/p13-rerank-0925-2257/ckpt/ce_bge_pool.pt\", \"s3\": \"s3://sagemaker-114872709824-ap-south-1/amc26-p30/results/$J\", \"deadline_s\": 6600}" \
      > data/work/p30/launch_last.log 2>&1; then
    echo "$J" > data/work/p30/JOB; echo "launched $J at $(date)"; exit 0
  fi
  grep -q ResourceLimitExceeded data/work/p30/launch_last.log || { echo "non-quota failure"; tail -5 data/work/p30/launch_last.log; exit 1; }
  sleep 60
done
