#!/usr/bin/env bash
# sub60: France rebuilt with P60's pooled Qwen (P38's LoRA warm-started on folds 1-3) through sub_clean's chain
# (build38 clean on sub54, hn .80, P48 veto r < .2 / P54 rescue r >= .99 on the new Qwen r), then P50's P55 + dup
# -> output/<S>_fr (P50's input: the clean chain only) and output/<S> (+ P55 + dup, sub54's candidate_pairs). S=sub60 default.
#   probes/P60-qwen-bigdata/run60.sh <pooled job> <a,b,c = mean-OOD France weights from p60.py gate>
set -euo pipefail
cd "$(dirname "$0")/../.."
J=$1 WQ=$2 PY=.venv/bin/python X=probes/P38-qwen4b-fr W=data/work/p60
RQ=${RQ:-$W/r/r_qwen60_pool_test_France.parquet} S=${S:-sub60}   # RQ / S: dry run on another r, output name
[[ -f $RQ ]] || $PY probes/P60-qwen-bigdata/p60.py fetch "$J" | tail -1
$PY $X/build38.py blend $RQ $WQ $W/blend_$S.parquet
T=$($PY $X/build38.py tpick $W/blend_$S.parquet | tail -1 | $PY -c "import json,sys; print(json.load(sys.stdin)['t'])")
echo "t (#9ab empty-rate match) = $T"
$PY $X/build38.py post $W/blend_$S.parquet "$T" $W/post_$S.parquet | tail -1 | cut -c1-600
$PY $X/build38.py clean $W/post_$S.parquet $W/blend_$S.parquet $RQ "$T" output/${S}_fr sub54 $S 0.8 0.2 0.99 | tail -1
$PY probes/P55-fr-combiner/p55.py build ${S}_fr _${S}_b v=0.6,scope=band,u=0.95 | tail -1 | cut -c1-330
$PY probes/P54-fresh-batch/build54.py _${S}_b $S | grep -E "dup_adds|cand_added|\"France\"" | tr -d '\n'; echo
ln -f output/_${S}_b/code.zip output/$S/code.zip && rm -rf output/_${S}_b
ln -f output/sub54/candidate_pairs.tsv output/$S/candidate_pairs.tsv
md5sum output/$S/*.tsv
