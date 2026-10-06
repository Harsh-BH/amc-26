#!/usr/bin/env bash
cd /home/harsh1/github-repos/amc-26-wt/p9-integrate
T=code/business_entity_resolution/src/test_pipeline.py
E="--ce data/work/p11/ce_s2P.pt --ce-base e5"
for args in "--recipe sub8c $E" "--recipe sub8c $E --unseen France --france sib" "--recipe sub8c $E --unseen France --france bge"; do
  echo "== $args"
  .venv/bin/python $T $args 2>&1 | grep -E "recipe |pipeline test|unseen|stacked|sibling|overridden|Error|assert|Traceback"
  echo "exit ${PIPESTATUS[0]}"
done
echo ALLDONE
