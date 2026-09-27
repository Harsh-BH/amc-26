#!/usr/bin/env bash
# status + last non-step log events of every P60 job (data/work/p60/jobs.txt). N=lines per job (default 3).
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export AWS_PROFILE=harsh AWS_DEFAULT_REGION=ap-south-1
B=s3://sagemaker-114872709824-ap-south-1/amc26-p60/results
while read -r j inst _; do
  s=$(aws sagemaker describe-training-job --training-job-name "$j" --query '[TrainingJobStatus,SecondaryStatus,BillableTimeInSeconds]' --output text 2>/dev/null)
  echo "== $j $inst: $s"
  [[ $s == Stopped* ]] && continue
  aws s3 cp "$B/$j/log.jsonl" - 2>/dev/null | grep -v '"ev": "step"' | tail -${N:-3} | cut -c1-300
  aws s3 cp "$B/$j/log.jsonl" - 2>/dev/null | grep '"ev": "step"' | tail -1 | cut -c1-200
done < data/work/p60/jobs.txt
