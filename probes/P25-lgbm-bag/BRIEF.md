# P25-lgbm-bag: a diverse LightGBM ensemble (bagging + boosting variants), trained and evaluated on AWS
The user asked for an ensemble of different LightGBMs using bagging/boosting techniques, all on AWS (credits fine;
the laptop is out of RAM because of other sessions — keep local work tiny). Own worktree; code under probes/P25-lgbm-bag/.
Reuse, don't rebuild: P22's AWS training + evaluation path lives in
/home/harsh1/github-repos/amc-26-wt/p22-ensemble/probes/P22-ensemble/ (p22.py: training jobs on the S3 train X under
amc26-p16/, an AWS eval job that reproduces #8c's holdout 0.98990 bit-identically in ~133 s). Read its code and
RESULT draft first; also probes/P16-aws-lgbm (8-fold LGBM +0.00014), probes/P4-match (m2_big params: lr 0.05, 127
leaves, ~4,930 trees, folds 2–5, early stop on fold 1). P22 found m2_big + P16 averaged is below the bar (too similar).
AWS: `--profile harsh` only. CPU instances (m7i/m6i/c7i 4xlarge…; check training quotas with service-quotas, never file
requests; P22 may hold one m7i.4xlarge). Run several variants in parallel jobs if quota allows. ≤ 3 h caps.
Variants (same 65 features, same folds 2–5, early stop on fold 1 unless noted) — maximise DIVERSITY:
- **Bagging**: 3–5 models with bagging_fraction 0.7 / bagging_freq 1 / feature_fraction 0.7 and different seeds;
  and/or models on bootstrap/disjoint S1 subsets of folds 2–9 (each sees different businesses).
- **Boosting variants**: DART (drop_rate 0.1), GOSS, extra_trees=True, a shallow model (num_leaves 31, more trees) and a
  deep one (num_leaves 255, min_data_in_leaf 200).
Evaluate on AWS with P22's eval path, on the #8c yardstick: logit-average (weights fit on holdout half 0) of m2_big +
variants → bge CE blend (P13 w refit on half 0) → P17 US stacker → report half 1 with paired SE vs #8c. Report each
variant alone, the equal-weight bag, and the fitted blend. Adopt bar: ≥ +0.0003 at ≥ 3 SE (India+US).
If adopted: test probabilities for India/US/France on #6's pruned rows (France via the #8c France recipe) so the master
can build a submission. RESULT.md with the table + $ spent. Memory 2 GB locally. Never end your turn to wait. Deadline 26 Sep 20:00.
