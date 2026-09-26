#!/usr/bin/env bash
# P25's eval job without Python (= P22's ev_launch.sh): CPU DLC, ml.m5.4xlarge on-demand, 1 h cap.
#   probes/P25-lgbm-bag/ev_launch.sh '<runs json>' v1 v2 ...
#   runs = [[["bag1","bag2"], "name", null | ["avg_all", ...]], ...]; the variants' preds come from their
#   training jobs (data/work/p25/job_<v>.txt). Job name -> data/work/p25/job_ev.txt.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RUNS=$1; shift
B=sagemaker-114872709824-ap-south-1 P=amc26-p25 NAME=p25-ev-$(date +%m%d-%H%M%S)
SRC=code/business_entity_resolution/src H=probes/P25-lgbm-bag P22=../p22-ensemble/probes/P22-ensemble
RES=$(for v in "$@"; do jq -n --arg v "$v" --arg j "$(cat data/work/p25/job_$v.txt)" \
        '{($v): "s3://sagemaker-114872709824-ap-south-1/amc26-p25/results/\($j)/"}'; done | jq -sc 'add')
T=$(mktemp -d)
cp sm_train.py $SRC/{rerank,stack,model,train_matcher}.py $P22/ev.py $H/ev25.py "$T"/
cp $H/solution_p25ev.py "$T"/solution.py
cp $P22/requirements_ev.txt "$T"/requirements.txt
sha1sum $P22/ev.py > data/work/p25/ev_py_sha1_$NAME.txt
tar czf "$T"/sourcedir.tar.gz -C "$T" sm_train.py solution.py ev.py ev25.py requirements.txt rerank.py stack.py model.py train_matcher.py
aws s3 cp "$T"/sourcedir.tar.gz s3://$B/$P/source/$NAME/sourcedir.tar.gz --profile harsh --region ap-south-1 --only-show-errors
jq -n --arg name "$NAME" --arg runs "$RUNS" --arg res "$RES" --arg src "s3://$B/$P/source/$NAME/sourcedir.tar.gz" \
      --arg s3 "s3://$B/$P/results/$NAME/" '{
  TrainingJobName: $name,
  AlgorithmSpecification: {TrainingImage: "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312",
                           TrainingInputMode: "File"},
  RoleArn: "arn:aws:iam::114872709824:role/AMCSageMakerExecutionRole",
  HyperParameters: {sagemaker_program: "\"sm_train.py\"", sagemaker_submit_directory: ($src | tojson),
                    runs: $runs, results: $res, s3: ($s3 | tojson)},
  InputDataConfig: [{ChannelName: "train", DataSource: {S3DataSource: {S3DataType: "S3Prefix",
                     S3Uri: "s3://sagemaker-114872709824-ap-south-1/amc26-p22/data/ev/", S3DataDistributionType: "FullyReplicated"}}}],
  OutputDataConfig: {S3OutputPath: "s3://sagemaker-114872709824-ap-south-1/amc26-p25/output/"},
  ResourceConfig: {InstanceType: "ml.m5.4xlarge", InstanceCount: 1, VolumeSizeInGB: 30},
  StoppingCondition: {MaxRuntimeInSeconds: 3600},
  Tags: [{Key: "project", Value: "amc26"}, {Key: "probe", Value: "P25"}]}' > "$T"/req.json
aws sagemaker create-training-job --cli-input-json file://"$T"/req.json --profile harsh --region ap-south-1 >/dev/null
echo "$NAME" > data/work/p25/job_ev.txt
echo "submitted $NAME results=$RES"
rm -rf "$T"
