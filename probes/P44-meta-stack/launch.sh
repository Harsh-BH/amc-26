#!/usr/bin/env bash
# One SageMaker CPU job per variant of config.json (P25's launch.sh request + an 'hn' channel). Only the variant name
# and the S3 result prefix go through hyperparameters.json; everything else is config.json inside the source tarball.
#   probes/P44-meta-stack/launch.sh rh:ml.m5.xlarge hn21:ml.m5.4xlarge m2hn:ml.m6i.4xlarge
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
H=probes/P44-meta-stack B=sagemaker-114872709824-ap-south-1 P=amc26-p44 TS=$(date +%m%d-%H%M)
A=(--profile harsh --region ap-south-1)
mkdir -p data/work/p44ms
aws s3 sync data/work/p44ms/hn "s3://$B/$P/data/hn/" "${A[@]}" --only-show-errors
T=$(mktemp -d)
cp sm_train.py $H/config.json "$T"/ && cp $H/solution_p44.py "$T"/solution.py && cp probes/P25-lgbm-bag/solution_p25.py "$T"/p25.py
printf 'lightgbm==4.7.0\npyarrow==25.0.1\nrapidfuzz==3.14.6\n' > "$T"/requirements.txt
tar czf "$T"/sourcedir.tar.gz -C "$T" sm_train.py solution.py p25.py config.json requirements.txt
SRC=s3://$B/$P/source/$TS/sourcedir.tar.gz
aws s3 cp "$T"/sourcedir.tar.gz "$SRC" "${A[@]}" --only-show-errors
ch() { jq -n --arg n "$1" --arg u "$2" '{ChannelName: $n, DataSource: {S3DataSource: {S3DataType: "S3Prefix",
          S3DataDistributionType: "FullyReplicated", S3Uri: $u}}}'; }
CH=$(jq -s . <(ch train "s3://$B/amc26-p16/data/v1/") <(ch test "s3://$B/amc26-p22/data/test/") <(ch hn "s3://$B/$P/data/hn/"))
for vi in "$@"; do
  v=${vi%%:*} inst=${vi#*:}
  NAME=p44-${v}-$TS
  jq -n --arg name "$NAME" --arg v "$v" --arg src "$SRC" --arg s3 "s3://$B/$P/results/$NAME/" --arg inst "$inst" \
        --argjson ch "$CH" '{
    TrainingJobName: $name,
    AlgorithmSpecification: {TrainingImage: "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312",
                             TrainingInputMode: "File",
                             MetricDefinitions: [{Name: "val_loss", Regex: "\\bval_loss=([0-9.eE+-]+)"}]},
    RoleArn: "arn:aws:iam::114872709824:role/AMCSageMakerExecutionRole",
    HyperParameters: {sagemaker_program: "\"sm_train.py\"", sagemaker_submit_directory: ($src | tojson),
                      name: ($v | tojson), s3: ($s3 | tojson)},
    InputDataConfig: $ch,
    OutputDataConfig: {S3OutputPath: "s3://sagemaker-114872709824-ap-south-1/amc26-p44/output/"},
    ResourceConfig: {InstanceType: $inst, InstanceCount: 1, VolumeSizeInGB: 30},
    StoppingCondition: {MaxRuntimeInSeconds: 7200},
    Tags: [{Key: "project", Value: "amc26"}, {Key: "probe", Value: "P44-meta-stack"}]}' > "$T/req_$v.json"
  aws sagemaker create-training-job --cli-input-json file://"$T/req_$v.json" "${A[@]}" >/dev/null
  echo "$NAME" > data/work/p44ms/job_$v.txt
  echo "submitted $NAME ($inst)"
done
rm -rf "$T"
