#!/usr/bin/env bash
W=/home/harsh1/github-repos/amc-26-wt/p10-candidates
cd /home/harsh1/github-repos/amc-26/data/student_resource
$W/.venv/bin/python utils/validate_submission.py --matching $W/output/p10check/matching_results.tsv \
    --candidate $W/output/p10check/candidate_pairs.tsv --test-dir dataset/test
echo VAL_EXIT $?
