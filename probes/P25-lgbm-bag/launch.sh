#!/usr/bin/env bash
# One SageMaker CPU job per LightGBM variant (variants.json), without Python (the laptop's memguard pauses probe
# Pythons). Same request as P22's p22-xgb job: PyTorch CPU DLC, sm_train.py -> solution_p25.py, on-demand, train =
# amc26-p16/data/v1/, test = amc26-p22/data/test/, MaxRuntime 3 h unless the variant sets max_run.
#   probes/P25-lgbm-bag/launch.sh bag1 bag2 ...        (job names -> data/work/p25/job_<variant>.txt)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
H=probes/P25-lgbm-bag V=$H/variants.json B=sagemaker-114872709824-ap-south-1 P=amc26-p25 TS=$(date +%m%d-%H%M)
mkdir -p data/work/p25
T=$(mktemp -d)
cp sm_train.py "$T"/ && cp $H/solution_p25.py "$T"/solution.py && cp $H/requirements_sm.txt "$T"/requirements.txt
tar czf "$T"/sourcedir.tar.gz -C "$T" sm_train.py solution.py requirements.txt
SRC=s3://$B/$P/source/$TS/sourcedir.tar.gz
aws s3 cp "$T"/sourcedir.tar.gz "$SRC" --profile harsh --region ap-south-1 --only-show-errors
for v in "$@"; do
  NAME=p25-${v//_/-}-$TS
  MAXRUN=$(jq -r --arg v "$v" '.[$v].max_run // 10800' $V)
  jq -n --arg name "$NAME" --arg v "$v" --arg src "$SRC" --arg s3 "s3://$B/$P/results/$NAME/" --argjson max "$MAXRUN" \
        --slurpfile vs $V '$vs[0] as $all | $all[$v] as $x | {
    TrainingJobName: $name,
    AlgorithmSpecification: {TrainingImage: "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312",
                             TrainingInputMode: "File",
                             MetricDefinitions: [{Name: "val_loss", Regex: "\\bval_loss=([0-9.eE+-]+)"}]},
    RoleArn: "arn:aws:iam::114872709824:role/AMCSageMakerExecutionRole",
    HyperParameters: {sagemaker_program: "\"sm_train.py\"", sagemaker_submit_directory: ($src | tojson),
                      name: ($v | tojson), folds: ($x.folds | tojson), rounds: ($x.rounds | tojson),
                      es: (($x.es // 100) | tojson), frac: (($x.frac // 1.0) | tojson),
                      deadline_s: (($max - 2400) | tojson), s3: ($s3 | tojson),
                      params: (($all._base + $x.params) | tojson)},
    InputDataConfig: [
      {ChannelName: "train", DataSource: {S3DataSource: {S3DataType: "S3Prefix", S3DataDistributionType: "FullyReplicated",
                       S3Uri: "s3://sagemaker-114872709824-ap-south-1/amc26-p16/data/v1/"}}},
      {ChannelName: "test", DataSource: {S3DataSource: {S3DataType: "S3Prefix", S3DataDistributionType: "FullyReplicated",
                       S3Uri: "s3://sagemaker-114872709824-ap-south-1/amc26-p22/data/test/"}}}],
    OutputDataConfig: {S3OutputPath: "s3://sagemaker-114872709824-ap-south-1/amc26-p25/output/"},
    ResourceConfig: {InstanceType: $x.inst, InstanceCount: 1, VolumeSizeInGB: 30},
    StoppingCondition: {MaxRuntimeInSeconds: $max},
    Tags: [{Key: "project", Value: "amc26"}, {Key: "probe", Value: "P25"}]}' > "$T/req_$v.json"
  aws sagemaker create-training-job --cli-input-json file://"$T/req_$v.json" --profile harsh --region ap-south-1 >/dev/null
  echo "$NAME" > data/work/p25/job_$v.txt
  cp "$T/req_$v.json" data/work/p25/req_$v.json
  echo "submitted $NAME ($(jq -r '.[$v].inst' --arg v "$v" $V), max $MAXRUN s)"
done
rm -rf "$T"
