#!/usr/bin/env bash
# One line per P25 job: SageMaker status, billable s, and the job's own status.json (state, iter, val loss).
#   probes/P25-lgbm-bag/status.sh [variant ...]      (default: every data/work/p25/job_*.txt)
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
B=sagemaker-114872709824-ap-south-1
vs=("$@"); [[ ${#vs[@]} -eq 0 ]] && vs=($(ls data/work/p25/job_*.txt | sed 's/.*job_//; s/\.txt//'))
for v in "${vs[@]}"; do
  j=$(cat data/work/p25/job_$v.txt)
  d=$(aws sagemaker describe-training-job --training-job-name "$j" --profile harsh --region ap-south-1 \
        --query '[TrainingJobStatus,SecondaryStatus,BillableTimeInSeconds,FailureReason]' --output text 2>&1 | tr '\t' ' ')
  s=$(aws s3 cp "s3://$B/amc26-p25/results/$j/status.json" - --profile harsh --region ap-south-1 2>/dev/null |
        jq -c '{state, iter, best_iter, val_loss, best_val_loss, trees, elapsed_s, peak_gb, deadline_hit}|with_entries(select(.value!=null))' 2>/dev/null)
  printf "%-8s %-28s %s %s\n" "$v" "$j" "$d" "$s"
done
