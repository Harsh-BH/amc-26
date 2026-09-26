#!/usr/bin/env bash
# p22.py evlaunch without Python: the laptop's memguard pauses any probe python while MemAvailable < 1.5 GB.
# Same request sm_launch.build_request makes (template: the p22-xgb job), CPU DLC, ml.m5.4xlarge on-demand, 1 h cap.
#   probes/P22-ensemble/ev_launch.sh '<runs json>' '<results json>'
#   e.g. ev_launch.sh '[[["p16"], "dry_p16"]]' '{}'
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RUNS=$1 RESULTS=${2:-"{}"}
B=sagemaker-114872709824-ap-south-1 P=amc26-p22 NAME=p22-ev-$(date +%m%d-%H%M%S)
SRC=code/business_entity_resolution/src H=probes/P22-ensemble
T=$(mktemp -d)
cp sm_train.py $SRC/{rerank,stack,model,train_matcher}.py $H/ev.py "$T"/
cp $H/solution_p22ev.py "$T"/solution.py
cp $H/requirements_ev.txt "$T"/requirements.txt
tar czf "$T"/sourcedir.tar.gz -C "$T" sm_train.py solution.py ev.py requirements.txt rerank.py stack.py model.py train_matcher.py
aws s3 cp "$T"/sourcedir.tar.gz s3://$B/$P/source/$NAME/sourcedir.tar.gz --profile harsh --region ap-south-1 --only-show-errors
jq -n --arg name "$NAME" --arg runs "$RUNS" --arg res "$RESULTS" --arg src "s3://$B/$P/source/$NAME/sourcedir.tar.gz" \
      --arg s3 "s3://$B/$P/results/$NAME/" --arg data "s3://$B/$P/data/ev/" '{
  TrainingJobName: $name,
  AlgorithmSpecification: {TrainingImage: "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312",
                           TrainingInputMode: "File"},
  RoleArn: "arn:aws:iam::114872709824:role/AMCSageMakerExecutionRole",
  HyperParameters: {sagemaker_program: "\"sm_train.py\"", sagemaker_submit_directory: ($src | tojson),
                    runs: $runs, results: $res, s3: ($s3 | tojson)},
  InputDataConfig: [{ChannelName: "train", DataSource: {S3DataSource: {S3DataType: "S3Prefix", S3Uri: $data,
                     S3DataDistributionType: "FullyReplicated"}}}],
  OutputDataConfig: {S3OutputPath: "s3://sagemaker-114872709824-ap-south-1/amc26-p22/output/"},
  ResourceConfig: {InstanceType: "ml.m5.4xlarge", InstanceCount: 1, VolumeSizeInGB: 30},
  StoppingCondition: {MaxRuntimeInSeconds: 3600},
  Tags: [{Key: "project", Value: "amc26"}]}' > "$T"/req.json
aws sagemaker create-training-job --cli-input-json file://"$T"/req.json --profile harsh --region ap-south-1 >/dev/null
echo "$NAME" > data/work/p22/job_ev.txt
echo "submitted $NAME"
rm -rf "$T"
