#!/usr/bin/env bash
# run each #31 variant, cmp both TSVs with main's staged folder, delete the output
R=/home/harsh1/github-repos/amc-26/output
run() {
  name=$1; shift
  .venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --no-validate --out output/$name "$@" \
    > data/work/p9/logs/$name.log 2>&1 || { echo "$name RUN FAILED"; return; }
  for f in matching_results.tsv candidate_pairs.tsv; do
    cmp -s output/$name/$f $R/$name/$f && echo "$name $f IDENTICAL" || echo "$name $f DIFFERS"
  done
  grep -E "^France: (key tier|bxneg|veto|cap|sibling)|France: 259" data/work/p9/logs/$name.log
  rm -rf output/$name
}
run sub31_t9818 --recipe sub31 --t-country France=0.9818
run sub31_veto --recipe sub31_veto
run sub31_small --recipe sub31_small
