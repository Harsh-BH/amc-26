#!/usr/bin/env bash
# One line per job every 60 s: status / secondary status / last training log line (CloudWatch). Stops when none is InProgress.
JOBS="$*"
while true; do
  live=0; line="$(date +%H:%M)"
  for j in $JOBS; do
    s=$(aws sagemaker describe-training-job --profile harsh --region ap-south-1 --training-job-name "$j" --query '[TrainingJobStatus,SecondaryStatus]' --output text 2>/dev/null | tr '\t' '/')
    [ "${s%%/*}" = "InProgress" ] && live=1
    last=""
    st=$(aws logs describe-log-streams --profile harsh --region ap-south-1 --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix "$j/" --query 'logStreams[0].logStreamName' --output text 2>/dev/null)
    if [ -n "$st" ] && [ "$st" != "None" ]; then
      last=$(aws logs get-log-events --profile harsh --region ap-south-1 --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name "$st" --limit 30 --query 'events[].message' --output text 2>/dev/null | tr '\t' '\n' | grep -E "valid_0|split|trees|done|files|Error|Traceback" | tail -1 | cut -c1-90)
    fi
    line="$line | ${j#p80-} $s${last:+ :: $last}"
  done
  echo "$line"
  [ $live = 0 ] && break
  sleep 60
done
