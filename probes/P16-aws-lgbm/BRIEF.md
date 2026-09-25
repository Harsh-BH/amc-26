# P16-aws-lgbm: retrain the LightGBM matcher on ALL training folds (2–9) on AWS
The user wants model training on AWS (credits are fine) and everything else on the laptop. Read first:
probes/P4-match/RESULT.md ("Improvement round": m2_big = folds 2–5 gave +0.00040 ± 0.00006 over fold 2 only),
code/business_entity_resolution/src/train_matcher.py (train_matcher.main(out, folds, frac, rounds) = big.py's recipe),
code/business_entity_resolution/README.md, sm_launch.py / sm_train.py / test_sm.py, probes/P13-aws-rerank/ (a working
SageMaker job: its solution file, requirements, S3 layout — reuse its patterns). Own worktree; code only under
probes/P16-aws-lgbm/.

AWS: ALWAYS `--profile harsh` / `AWS_PROFILE=harsh` (default is another account). Account 114872709824, ap-south-1,
bucket sagemaker-114872709824-ap-south-1 (prefix amc26-p16/), role AMCSageMakerExecutionRole. CPU training quota
is ~15 on-demand for most types; use a big-RAM CPU instance (e.g. ml.m5.4xlarge 64 GB or ml.r5.4xlarge 128 GB —
check the quota with service-quotas first, never file a request). Credits are fine; still set MaxRuntime (≤ 4 h).
Look up APIs with ~/.claude/skills/aws-docs/awsdoc.py before calling them.

Task:
1. Upload the train-side inputs the trainer needs (p4/X/train_* pair features, labels/folds — whatever
   train_matcher.main reads) to S3 once.
2. Rehearse the exact job locally on a 1% slice (CPU), then launch ONE SageMaker job: same LightGBM params and
   early stopping as m2_big, folds **2–9** (8 folds, ~25M pairs), early-stopped on fold 1. Save the booster to S3.
3. Back on the laptop: score the holdout and evaluate on the SAME yardstick as #6/#7 (full-corpus holdout, P5 report
   half, CE blend refit on the tune half, pruner kept). Report Δ vs m2_big with SE. Adopt bar: ≥ +0.0002 at ≥ 3 SE
   (zero-complexity change, like m2_big).
4. If adopted: note that the pruner and the CE band/blend were fit on m2_big's p — check whether they need a refit
   (the pruner uses X columns, not p, so it should not; the CE blend weights should be refit on the tune half).
   Produce test scores for the new booster (laptop CPU, or AWS if faster) → data/work/p16/prob_test_{c}.parquet.
Memory cap 2 GB locally incl. CLI (heavy work runs on AWS). Never end your turn to wait; poll with bounded waits.
Ledger every run; report the $ spent in RESULT.md.
