#!/usr/bin/env bash
# last N lines of a P26 job's CloudWatch log:  logs.sh <tag> [N]
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
j=$(cat data/work/p26/job_$1.txt)
s=$(aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix "$j/" \
    --profile harsh --region ap-south-1 --query 'logStreams[0].logStreamName' --output text)
aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name "$s" --profile harsh \
    --region ap-south-1 --no-start-from-head --limit ${2:-8} --query 'events[].message' --output text | tr '\t' '\n' | cut -c1-700
