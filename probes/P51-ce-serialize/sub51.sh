#!/usr/bin/env bash
# sub51.sh <r.parquet|bge> <a,b,c> <tag> [BASE=output/sub44r]: France = #46_on43's chain (build38 blend -> tpick ->
# P31 post (veto 0.1) -> on43 -> P46 swap lens) with the given band r + OOD weights; India/US lines = BASE's.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=$1 WT=$2 TAG=$3 BASE=${4:-output/sub44r} PY=.venv/bin/python X=probes/P38-qwen4b-fr W=data/work/p51/chain
mkdir -p $W
$PY $X/build38.py blend "$R" "$WT" $W/blend_$TAG.parquet | tail -1
T=$($PY $X/build38.py tpick $W/blend_$TAG.parquet | tail -1 | $PY -c "import json,sys; print(json.load(sys.stdin)['t'])")
echo "t (#9ab empty-rate match) = $T"
$PY $X/build38.py post $W/blend_$TAG.parquet "$T" $W/post_$TAG.parquet | tail -1 | cut -c1-400
$PY $X/build38.py on43 $W/post_$TAG.parquet "$T" $W/on43_$TAG | tail -1 | cut -c1-600
$PY probes/P46-noise-word-tier/p46.py build $W/on43_$TAG $W/on46_$TAG lens | tail -2 | cut -c1-400
$PY probes/P51-ce-serialize/splice.py $W/on46_$TAG "$BASE" output/sub51_$TAG
