# P22-ensemble: does a second GBDT family (XGBoost / CatBoost) averaged with m2_big help? (train on AWS)
Read: probes/P4-match/RESULT.md, probes/P16-aws-lgbm/ (its SageMaker CPU job pattern, S3 layout; the train X is already
in S3 under amc26-p16/), train_matcher.py, probes/P11-ce-big (ensembles of CEs gave ≤ +0.00002), probes/P5-decide.
The user asked whether we tried ensembles / XGBoost. We have: LightGBM + CE blend, CE ensembles (no gain), P5 stack
(+0.0001), P17 residual stacker (US +0.00055). Never a second GBDT family. Own worktree; code under probes/P22-ensemble/.
AWS: `--profile harsh` only (default is another account); ml.m7i.4xlarge on-demand like P16; credits are fine; ≤ 3 h cap.
Licences: xgboost Apache-2.0, catboost Apache-2.0 — confirm.
Task: train XGBoost (hist) and/or CatBoost on the same 65 features, folds 2–5 (same as m2_big), early-stopped on fold 1,
on SageMaker. Then on the laptop: average with m2_big in logit space (weights fit on holdout half 0), re-blend the bge CE
on top (P13 w refit), evaluate on the #8c yardstick (run.py holdout, report half, paired SE). Adopt bar: ≥ +0.0003 at
≥ 3 SE on India+US. If adopted, produce test probabilities for India/US (and France) on #6's pruned rows. Also report
the plain rank-average variant. Memory 2 GB locally. Never end your turn to wait. Deadline 26 Sep 19:00.
