#!/bin/bash
# wait until every expected r_ind file is on S3 (or a job left InProgress), then print what exists
B=s3://sagemaker-114872709824-ap-south-1/amc26-p73/results
want="$*"   # e.g. p73-bge-loco-0927-1516/India_bge/r/r_ind_bge_fb_US_India.parquet ...
while :; do
  miss=0; for w in $want; do aws s3 ls $B/$w --profile harsh --region ap-south-1 >/dev/null 2>&1 || miss=1; done
  [ $miss = 0 ] && break
  run=0; for j in $(for w in $want; do echo ${w%%/*}; done | sort -u); do
    s=$(aws sagemaker describe-training-job --training-job-name $j --profile harsh --region ap-south-1 --query TrainingJobStatus --output text); [ "$s" = InProgress ] && run=1; done
  [ $run = 0 ] && break
  sleep 60
done
for w in $want; do aws s3 ls $B/$w --profile harsh --region ap-south-1 >/dev/null 2>&1 && echo "OK $w" || echo "MISSING $w"; done
