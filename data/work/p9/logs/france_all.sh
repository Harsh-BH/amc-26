#!/usr/bin/env bash
# every France variant on the #8c base -> output/<r>, cmp both files against the staged submission, then drop the copy
cd /home/harsh1/github-repos/amc-26-wt/p9-integrate
for r in ${RECIPES_TO_RUN:-sub9s sub9b sub9a sub9d}; do
  t0=$(date +%s)
  mkdir -p output/$r
  .venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --recipe $r --out output/$r > data/work/p9/logs/test_$r.log 2>&1
  echo "$r run exit $? in $(( $(date +%s) - t0 ))s"
  for f in matching_results.tsv candidate_pairs.tsv; do
    cmp -s output/$r/$f /home/harsh1/github-repos/amc-26/output/$r/$f && echo "$r $f BYTE-IDENTICAL" || echo "$r $f DIFFERS"
  done
  .venv/bin/python probes/P9-integrate/diff_sub4.py output/$r /home/harsh1/github-repos/amc-26/output/$r log 2>&1 | head -2
done
echo ALLDONE
