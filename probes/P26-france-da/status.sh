#!/usr/bin/env bash
# one line per P26 job: status, secondary status, elapsed; + the job's last S3 status.json state
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
for f in data/work/p26/job_*.txt; do
  j=$(cat "$f")
  d=$(aws sagemaker describe-training-job --training-job-name "$j" --profile harsh --region ap-south-1 \
      --query '[TrainingJobStatus,SecondaryStatus,TrainingTimeInSeconds,FailureReason]' --output text 2>/dev/null)
  st=$(aws s3 cp "s3://sagemaker-114872709824-ap-south-1/amc26-p26/results/$j/status.json" - --profile harsh --region ap-south-1 2>/dev/null | jq -r '.state // empty' 2>/dev/null)
  echo "$j | $d | $st"
done
