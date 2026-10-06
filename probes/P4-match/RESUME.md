Master ruling: ADOPT m2_big (folds 2-5). Your +0.00040 (6.7 SE) is certain, and the change adds no complexity (same
recipe, only more training folds), so the 0.001 bar — meant to stop complexity for noisy gains — does not apply.
Produce the test files exactly as your RESULT.md describes:
1. `big.py test` -> data/work/p4/prob_test_{France,India,US}_big.parquet (same schema as the shipped prob_test_* files;
   argmax over ALL S1 of the country; do not overwrite the shipped files).
2. `flock /tmp/amc-gpu.lock big.py blendtest India` and `... US` -> prob_test_{India,US}_big_blend.parquet using the
   refit weights w=[0.678, 0.715, 0.338] and band 0.02<p<0.98 (reuse cached CE scores where the pair was already scored).
3. Run P6's France rule on the big model's test probabilities: `.venv/bin/python probes/P6-france/fr_threshold.py
   'data/work/p4/prob_test_{c}_big.parquet' 0.69` (t* 0.69 for the big model) and report t_France.
4. Log ledger rows (pairs, matches/S1, empty share per country at the thresholds), update RESULT.md, then stop.
Cap 4 GB incl. the CLI, swap allowed.
