#!/usr/bin/env bash
set -e
cd /home/harsh1/github-repos/amc-26-wt/p10-candidates
.venv/bin/python code/business_entity_resolution/src/submit_probs.py --probs 'data/work/p10/prob_test_{country}.parquet' \
    --out output/p10check --t 0.69 --t-country France=0.78
cd data/student_resource
../../.venv/bin/python utils/validate_submission.py --matching ../../output/p10check/matching_results.tsv \
    --candidate ../../output/p10check/candidate_pairs.tsv --test-dir dataset/test
echo E2E_DONE
