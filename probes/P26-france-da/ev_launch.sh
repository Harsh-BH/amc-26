#!/usr/bin/env bash
# #8c-yardstick eval of P26 matchers = P25's ev_launch.sh (P22 ev.py + P25 ev25.py + solution_p25ev.py, unchanged) with
# the variants' holdout preds from P26 score jobs:  ev_launch.sh '<runs json>' tag1 tag2 ...   (tag = job_<tag>.txt)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RUNS=$1; shift
B=sagemaker-114872709824-ap-south-1 P=amc26-p26 NAME=p26-ev-$(date +%m%d-%H%M%S)
SRC=code/business_entity_resolution/src P22=../p22-ensemble/probes/P22-ensemble P25=../p25-lgbm-bag/probes/P25-lgbm-bag
RES=$(for v in "$@"; do jq -n --arg v "$v" --arg j "$(cat data/work/p26/job_$v.txt)" \
        '{($v): "s3://sagemaker-114872709824-ap-south-1/amc26-p26/results/\($j)/"}'; done | jq -sc 'add')
T=$(mktemp -d)
cp sm_train.py $SRC/{rerank,stack,model,train_matcher}.py $P22/ev.py $P25/ev25.py "$T"/
cp $P25/solution_p25ev.py "$T"/solution.py
cp $P22/requirements_ev.txt "$T"/requirements.txt
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
  OutputDataConfig: {S3OutputPath: "s3://sagemaker-114872709824-ap-south-1/amc26-p26/output/"},
  ResourceConfig: {InstanceType: "ml.m6i.4xlarge", InstanceCount: 1, VolumeSizeInGB: 30},
  StoppingCondition: {MaxRuntimeInSeconds: 3600},
  Tags: [{Key: "project", Value: "amc26"}, {Key: "probe", Value: "P26"}]}' > "$T"/req.json
aws sagemaker create-training-job --cli-input-json file://"$T"/req.json --profile harsh --region ap-south-1 >/dev/null
echo "$NAME" > data/work/p26/job_ev.txt
echo "submitted $NAME results=$RES"
rm -rf "$T"
