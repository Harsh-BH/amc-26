#!/usr/bin/env bash
# Stage A pruner-zoo job: probes/P45-ens-stages/launch_a.sh <name> <instance> <members.json>
# hp = {"members": <members.json>, "s3": results prefix} travel in hyperparameters.json (never argv).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
H=probes/P45-ens-stages B=sagemaker-114872709824-ap-south-1 P=amc26-p45 TS=$(date +%m%d-%H%M) NAME=p45-$1-$TS
T=$(mktemp -d)
cp sm_train.py "$T"/ && cp $H/solution_p45a.py "$T"/solution.py && cp probes/P25-lgbm-bag/requirements_sm.txt "$T"/requirements.txt && cp data/work/p10/pruner.txt data/work/p10/pruner_big.txt "$T"/
tar czf "$T"/sourcedir.tar.gz -C "$T" sm_train.py solution.py requirements.txt pruner.txt pruner_big.txt
SRC=s3://$B/$P/source/$NAME/sourcedir.tar.gz
aws s3 cp "$T"/sourcedir.tar.gz "$SRC" --profile harsh --region ap-south-1 --only-show-errors
jq -n --arg name "$NAME" --arg src "$SRC" --arg s3 "s3://$B/$P/results/$NAME/" --arg inst "$2" --slurpfile mem "$3" '{
  TrainingJobName: $name,
  AlgorithmSpecification: {TrainingImage: "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312",
                           TrainingInputMode: "File"},
  RoleArn: "arn:aws:iam::114872709824:role/AMCSageMakerExecutionRole",
  HyperParameters: {sagemaker_program: "\"sm_train.py\"", sagemaker_submit_directory: ($src | tojson),
                    members: ($mem[0] | tojson), s3: ($s3 | tojson)},
  InputDataConfig: [{ChannelName: "train", DataSource: {S3DataSource: {S3DataType: "S3Prefix", S3DataDistributionType: "FullyReplicated",
                     S3Uri: "s3://sagemaker-114872709824-ap-south-1/amc26-p16/data/v1/"}}}],
  OutputDataConfig: {S3OutputPath: "s3://sagemaker-114872709824-ap-south-1/amc26-p45/output/"},
  ResourceConfig: {InstanceType: $inst, InstanceCount: 1, VolumeSizeInGB: 30},
  StoppingCondition: {MaxRuntimeInSeconds: 5400},
  Tags: [{Key: "project", Value: "amc26"}, {Key: "probe", Value: "P45"}]}' > "$T/req.json"
aws sagemaker create-training-job --cli-input-json file://"$T/req.json" --profile harsh --region ap-south-1 >/dev/null
mkdir -p data/work/p45 && echo "$NAME" > data/work/p45/job_a_$1.txt
echo "submitted $NAME ($2)"; rm -rf "$T"
