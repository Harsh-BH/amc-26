#!/usr/bin/env bash
cd /home/harsh1/github-repos/amc-26-wt/p9-integrate
R=/home/harsh1/github-repos/amc-26/output
for v in 975:0.975 990:0.99; do
  n=${v%%:*}; t=${v##*:}; o=output/sub9ab_t$n; mkdir -p $o
  .venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --t-country France=$t --out $o > data/work/p9/logs/test9ab_t$n.log 2>&1
  echo "t$n exit $?; $(grep -c PASS data/work/p9/logs/test9ab_t$n.log) PASS"
  for f in matching_results.tsv candidate_pairs.tsv; do cmp -s $o/$f $R/sub9ab_t$n/$f && echo "t$n $f BYTE-IDENTICAL" || echo "t$n $f DIFFERS"; done
  rm -rf $o
done
mkdir -p output/sub9ab_cap
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --cap 5,6 --out output/sub9ab_cap > data/work/p9/logs/test9ab_cap.log 2>&1
echo "cap exit $?; $(grep -c PASS data/work/p9/logs/test9ab_cap.log) PASS"
grep -E "cap 5 S2" data/work/p9/logs/test9ab_cap.log
.venv/bin/python probes/P9-integrate/diff_sub4.py output/sub9ab_cap $R/sub9ab 2>&1 | head -2
echo ALLDONE
