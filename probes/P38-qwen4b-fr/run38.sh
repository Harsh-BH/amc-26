#!/usr/bin/env bash
# sub38 (#37_gc70 with only the France CE swapped) from the pooled job's France band r, then the official validator
# (--check-ids) on SageMaker. sub38_full / sub38_clean: build38.py full / clean (see RESULT.md).
#   probes/P38-qwen4b-fr/run38.sh <pooled job name>
set -euo pipefail
cd "$(dirname "$0")/../.."
J=$1 PY=.venv/bin/python X=probes/P38-qwen4b-fr W=data/work/p38
B=s3://sagemaker-114872709824-ap-south-1
WQ=0.8943,0.3729,-0.0521                    # Qwen's mean-OOD France weights (p38.py gate, mean mode)
$PY $X/p38.py fetch "$J" | tail -1
$PY $X/build38.py blend $W/r/r_qwen_pool_test_France.parquet $WQ $W/blend_qwen.parquet
T=$($PY $X/build38.py tpick $W/blend_qwen.parquet | tail -1 | $PY -c "import json,sys; print(json.load(sys.stdin)['t'])")
echo "t (#9ab empty-rate match) = $T"
$PY $X/build38.py post $W/blend_qwen.parquet "$T" $W/post_qwen.parquet | tail -1 | cut -c1-600
$PY $X/build38.py sub $W/post_qwen.parquet "$T" output/sub38
export AWS_PROFILE=harsh
aws s3 rm $B/amc26-p38/val/matching_sub37_gc70_control.tsv --region ap-south-1 --only-show-errors || true
for s in sub38; do
  aws s3 cp output/$s/matching_results.tsv $B/amc26-p38/val/matching_$s.tsv --region ap-south-1 --only-show-errors
done
N=p38-val-$(date +%m%d-%H%M)
$PY sm_launch.py --profile harsh --solution $X/solution_val.py --no-spot --data-s3 $B/amc26-p38/val/ \
  --test-s3 $B/amc26-p43/test/ --instance ml.m5.xlarge --prefix amc26-p38 --name "$N" --max-run 3600 | tail -2 | head -1
echo "$N" >> $W/JOBS
