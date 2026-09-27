#!/usr/bin/env bash
# Score the new CE (ckpt of a P59 pool job) on every holdout argmax pair, every test India/US argmax pair and the
# France CE pair set (data/work/p50/r_bge_france_npz rows = P42 france.npz pairs).
#   score59.sh <ckpt s3 uri> <tag>      (one job per GPU type; test = 2 contiguous slices per country, P45's solution_p45c)
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CK=$1 TAG=$2 TS=$(date +%m%d-%H%M) B=s3://sagemaker-114872709824-ap-south-1/amc26-p59
L() { .venv/bin/python probes/P59-ce-bigdata/launch59.py "$@" 2>&1 | tail -1; sleep 2; }
HP() { echo "{\"phases\": [\"bge_ckpt\"], \"ckpt\": \"$CK\", \"s3\": \"$B/results/$1\", \"deadline_s\": 6000${2:-}}"; }
n=p59-ho-$TAG-$TS;  L $n ${HO_INST:-ml.g6e.xlarge}  $B/data/ho/ "$(HP $n)"
S=probes/P45-ens-stages/solution_p45c.py
for spec in "in0 India 0 2 ${IN0_INST:-ml.g6e.2xlarge}" "in1 India 1 2 ${IN1_INST:-ml.g6e.4xlarge}" "us0 US 0 2 ${US0_INST:-ml.g6e.8xlarge}" "us1 US 1 2 ${US1_INST:-ml.g5.4xlarge}" "fr France 0 1 ${FR_INST:-ml.g6.4xlarge}"; do
  set -- $spec
  n=p59-te$1-$TAG-$TS; L $n $5 $B/data/cetest/ "$(HP $n ", \"country\": \"$2\", \"part\": [$3, $4]")" $S
done
