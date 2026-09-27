#!/usr/bin/env bash
# P26 SageMaker CPU jobs (bash + aws cli + jq, = P25's launch.sh): PyTorch CPU DLC, sm_train.py -> solution_p26.py.
# train = amc26-p16/data/v1/ (P4 train X), test = amc26-p26/data/v1/ (meta npz, IDF-fixed France X, keys, m2_big).
#   probes/P26-france-da/launch.sh <tag> <instance> '<hp json>'      (job name -> data/work/p26/job_<tag>.txt)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
TAG=$1 INST=$2 HP=$3
H=probes/P26-france-da B=sagemaker-114872709824-ap-south-1 P=amc26-p26 TS=$(date +%m%d-%H%M)
NAME=p26-${TAG//_/-}-$TS
MAXRUN=${MAXRUN:-10800}
T=$(mktemp -d)
cp sm_train.py "$T"/ && cp $H/solution_p26.py "$T"/solution.py && cp $H/requirements_sm.txt "$T"/requirements.txt
tar czf "$T"/sourcedir.tar.gz -C "$T" sm_train.py solution.py requirements.txt
SRC=s3://$B/$P/source/$NAME/sourcedir.tar.gz
aws s3 cp "$T"/sourcedir.tar.gz "$SRC" --profile harsh --region ap-south-1 --only-show-errors
jq -n --arg name "$NAME" --arg src "$SRC" --arg s3 "s3://$B/$P/results/$NAME/" --arg inst "$INST" \
      --argjson max "$MAXRUN" --argjson hp "$HP" '{
  TrainingJobName: $name,
  AlgorithmSpecification: {TrainingImage: "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312",
                           TrainingInputMode: "File",
                           MetricDefinitions: [{Name: "val_loss", Regex: "\\bval_loss=([0-9.eE+-]+)"}]},
  RoleArn: "arn:aws:iam::114872709824:role/AMCSageMakerExecutionRole",
  HyperParameters: ({sagemaker_program: "\"sm_train.py\"", sagemaker_submit_directory: ($src | tojson),
                     s3: ($s3 | tojson), deadline_s: (($max - 2400) | tojson)}
                    + ($hp | with_entries(.value |= tojson))),
  InputDataConfig: [
    {ChannelName: "train", DataSource: {S3DataSource: {S3DataType: "S3Prefix", S3DataDistributionType: "FullyReplicated",
                     S3Uri: "s3://sagemaker-114872709824-ap-south-1/amc26-p16/data/v1/"}}},
    {ChannelName: "test", DataSource: {S3DataSource: {S3DataType: "S3Prefix", S3DataDistributionType: "FullyReplicated",
                     S3Uri: "s3://sagemaker-114872709824-ap-south-1/amc26-p26/data/v1/"}}}],
  OutputDataConfig: {S3OutputPath: "s3://sagemaker-114872709824-ap-south-1/amc26-p26/output/"},
  ResourceConfig: {InstanceType: $inst, InstanceCount: 1, VolumeSizeInGB: 40},
  StoppingCondition: {MaxRuntimeInSeconds: $max},
  Tags: [{Key: "project", Value: "amc26"}, {Key: "probe", Value: "P26"}]}' > "$T/req.json"
aws sagemaker create-training-job --cli-input-json file://"$T/req.json" --profile harsh --region ap-south-1 >/dev/null
mkdir -p data/work/p26
echo "$NAME" > data/work/p26/job_$TAG.txt
cp "$T/req.json" data/work/p26/req_$TAG.json
echo "submitted $NAME ($INST, max $MAXRUN s)"
rm -rf "$T"
