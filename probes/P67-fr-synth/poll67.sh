#!/usr/bin/env bash
# Status + the newest log.jsonl events of each P67 job (bge: <job>/<k>/log.jsonl, qwen: <job>/q{pre,syn}/log.jsonl).
export AWS_PROFILE=harsh AWS_DEFAULT_REGION=ap-south-1
B=s3://sagemaker-114872709824-ap-south-1/amc26-p67/results
for j in "$@"; do
  aws sagemaker describe-training-job --training-job-name "$j" --query '[TrainingJobName,TrainingJobStatus,SecondaryStatus,FailureReason]' --output text | cut -c1-400
  for f in $(aws s3 ls --recursive "$B/$j/" | awk '{print $4}' | grep 'log.jsonl$'); do
    aws s3 cp "s3://sagemaker-114872709824-ap-south-1/$f" - 2>/dev/null | grep -v '"ev": "step"\|tokens\|micro_ok' | tail -3 | cut -c1-300 | sed "s|^|  ${f#amc26-p67/results/} |"
  done
done
