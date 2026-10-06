cd /home/harsh1/github-repos/amc-26-wt/p13-aws-rerank
.venv/bin/python code/business_entity_resolution/src/submit_probs.py --probs 'data/work/p13/prob_test_{country}_blend.parquet' --out output/p13_sub --t 0.71 --t-country France=0.74
echo "SUBMIT_EXIT $?"
cd data/student_resource && python3 utils/validate_submission.py --matching ../../output/p13_sub/matching_results.tsv --candidate ../../output/p13_sub/candidate_pairs.tsv --test-dir dataset/test
echo "VALIDATE_EXIT $?"
