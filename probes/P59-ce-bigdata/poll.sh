#!/usr/bin/env bash
# status + last log events of every P59 job (data/work/p59/jobs.txt)
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
B=s3://sagemaker-114872709824-ap-south-1/amc26-p59/results
while read -r j inst; do
  s=$(aws sagemaker describe-training-job --training-job-name "$j" --profile harsh --region ap-south-1 \
      --query '[TrainingJobStatus,SecondaryStatus,BillableTimeInSeconds]' --output text 2>/dev/null)
  echo "== $j $inst: $s"
  aws s3 cp "$B/$j/log.jsonl" - --profile harsh --region ap-south-1 2>/dev/null | grep -v '"ev": "step"' | tail -${N:-3} | cut -c1-330
done < data/work/p59/jobs.txt
