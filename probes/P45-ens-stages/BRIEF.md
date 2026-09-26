# P45-ens-stages: ensembles (bagging / boosting / stacking) at EVERY stage — searcher, matcher, reranker — tested rigorously
User request: apply tree ensembles everywhere. Prior art (read first, do not repeat): P22-ensemble and P25-lgbm-bag
(averaging LGBM/XGB/CatBoost and bagging the matcher: dropped), P10 (pruner τ trade-off; more candidates cost more than
they recover, −0.00024), P13/P38 (bge vs Qwen; bge+Qwen logit-mean was below Qwen alone), P34 (a10 residual stacker =
current India/US). What is NOT yet tested and is your scope:
A. Searcher/pruner: a bagged / multi-seed / diverse-feature pruner ensemble at the SAME candidates per S1 (4.83; small
   version 4.42). Measure candidate recall and downstream F0.5 on the holdout. Smaller-set-at-equal-recall also counts
   (candidate-set size ranks teams beyond the LB).
B. Matcher: STACKED (not averaged) diverse boosters — LightGBM (gbdt, goss, dart), XGBoost, CatBoost, ExtraTrees/RF —
   trained on the same non-holdout folds, combined by a meta-learner fit on holdout half 0 and reported on half 1 + swap,
   and as extra columns in the a10 stacker. India/US gate: ≥ +0.00015 at ≥ 3 SE, neither < −0.00005.
C. Reranker: multi-seed / multi-checkpoint CE ensembles only if cheap (reuse P38's saved Qwen LoRA LOCO checkpoints and
   bge; score-level stacking with a learned combiner rather than a plain mean). France: LOCO both directions.
Own worktree, code in probes/P45-ens-stages/. Coordinate with P44 (same idea family): don't duplicate its 2nd-GBDT +
logistic meta; P45 owns multi-model boosters, the pruner and the CE side.
Compute: train the model zoo as parallel SageMaker CPU jobs (sm_launch.py; ml.m5.4xlarge / m6i.4xlarge $0.97/h 4 each,
ml.r6i.4xlarge ≈$1.25/h 4, c6i.4xlarge $0.82/h 2; prefix amc26-p45/; smoke first; hyperparameters from a config file,
never argv). GPU only if C needs it (then single g6e/g5, coordinate with P38 via its RESULT.md; quota 1 per type).
Budget $40. If anything passes, build output/sub45 on output/sub43_r90 (France byte-identical unless C passes the France
gate), validator PASS --check-ids, SUBMISSIONS.md row with expected LB.
RESULT.md with a table per stage. Local memory 2 GB. DISK ~3 GB. Never end your turn to wait.
Deadline: verdicts by 27 Sep 08:00 IST, sub45 (if any) by 10:30.
