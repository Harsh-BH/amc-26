#!/usr/bin/env bash
# One line per new event for each job in $@: SageMaker status changes, new log.jsonl events from S3 (step lines
# thinned to every 500th), and FailureReason / Traceback lines from CloudWatch when a job fails. Exits when all end.
export AWS_PROFILE=harsh AWS_DEFAULT_REGION=ap-south-1
B=s3://sagemaker-114872709824-ap-south-1/amc26-p38/results
declare -A ST SEEN
while :; do
  alive=0
  for j in "$@"; do
    s=$(aws sagemaker describe-training-job --training-job-name "$j" --query '[TrainingJobStatus,SecondaryStatus]' --output text 2>/dev/null || echo "? ?")
    if [[ "$s" != "${ST[$j]:-}" ]]; then
      echo "$j STATUS $s"; ST[$j]=$s
      [[ "$s" == Failed* || "$s" == Stopped* ]] && aws sagemaker describe-training-job --training-job-name "$j" \
        --query FailureReason --output text | cut -c1-600 | sed "s/^/$j REASON /"
    fi
    n=${SEEN[$j]:-0}
    if aws s3 cp "$B/$j/log.jsonl" /tmp/p38_$j.jsonl --only-show-errors 2>/dev/null; then
      tail -n +$((n + 1)) /tmp/p38_$j.jsonl | grep -vE '"ev": "(step|tokens|micro_ok)"|"ev": "step".*"step": [0-9]*[1-46-9]00,' \
        | cut -c1-260 | sed "s/^/$j /"
      tail -n +$((n + 1)) /tmp/p38_$j.jsonl | grep -E '"ev": "step", "step": [0-9]*[05]00,' | cut -c1-160 | sed "s/^/$j /"
      SEEN[$j]=$(wc -l < /tmp/p38_$j.jsonl)
    fi
    case "$s" in
      Failed*|Stopped*|Completed*) ;;
      *) alive=1 ;;
    esac
  done
  [[ $alive == 0 ]] && break
  sleep 60
done
