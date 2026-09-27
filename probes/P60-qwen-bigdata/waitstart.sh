#!/usr/bin/env bash
# exits (printing the job) as soon as any listed job leaves Pending/Starting, or after $2 s (default 1200)
export AWS_PROFILE=harsh AWS_DEFAULT_REGION=ap-south-1
end=$(( $(date +%s) + ${2:-1200} ))
while (( $(date +%s) < end )); do
  for j in $1; do
    s=$(aws sagemaker describe-training-job --training-job-name "$j" --query SecondaryStatus --output text 2>/dev/null)
    [[ -n $s && $s != Pending && $s != Starting ]] && { echo "$(date +%H:%M:%S) $j $s"; exit 0; }
  done
  sleep 30
done
echo "timeout"
