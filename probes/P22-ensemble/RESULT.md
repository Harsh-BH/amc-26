# P22-ensemble — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- DRAFT (13:25). XGBoost (hist) and CatBoost are training on SageMaker now: m2_big's data (folds 2–5, 12,595,232 pairs), early stopping on fold 1 (793,402 pairs), two parallel ml.m7i.4xlarge jobs launched 13:00.
- The laptop evaluator reproduces #8c's holdout pipe row exactly: all 0.98990, half 0 0.99024, half 1 0.98956 (India 0.99048, US 0.98951).
- No verdict yet. From 13:20 the laptop's memguard paused every probe Python I started (MemAvailable 0.8–1.5 GB,
  load ~40). So the evaluation moved to a CPU SageMaker job (`solution_p22ev.py`, launched by `ev_launch.sh` without Python).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Pending.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| rehearse-1pct-xgb-cat | train-1pct-slice | sm_train.py → solution.py, both families; every key scored; S3 writes; deadline stop keeps the best iteration | keep |
| sm-jobs-launch | aws | p22-xgb-0926-1300, p22-cat-0926-1300 on ml.m7i.4xlarge on-demand | — |
| (repro) | holdout-dense | ev.py control = #8c pipe row to 4 dp (0.9899 / 0.9902 / 0.9896); the AWS eval job gives the identical float | — |
| M2-xgb-train | train | 4,656 trees (early-stopped), fold-1 val logloss 0.023166 (m2_big 0.02285); 2,538 s billable; peak 16.6 GB | keep (pair level) |
| eval-m2big+m2p16-avg-8c | holdout-dense-half1 | dry run with P16's m2_p16 as the second model: lr +0.00007 ± 0.00006, avg +0.00006 ± 0.00004, rank +0.00006 ± 0.00004 vs #8c | drop |

## What failed      (approaches tried and dropped, with the number that killed them)
- Pending.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `solution_p22.py` (SageMaker job), `p22.py` (tiny / rehearse / upload / launch / status / fetch), `ev.py` (frame / repro / eval).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Licences confirmed from the installed wheels' metadata: xgboost-cpu 3.4.1 Apache-2.0, catboost 1.2.10 Apache-2.0.

## Open questions   (for the master; each with your best guess)
- Pending.

## Master conclusion (14:35, from the AWS eval jsons; the probe hit the usage limit at 14:14)
Every blend of m2_big with XGBoost and/or CatBoost is ≈ 0 or negative vs #8c on the report half (P13 bge blend refit +
P17 US stacker on top): best `avg_xgb` refit +0.00003 ± 0.00004, `avg_xgb+cat` ship +0.000005 ± 0.00004; XGBoost alone
−0.00010; CatBoost (capped at 8000 rounds, val 0.02436) no better. m2_big + P16 8-fold LGBM: +0.00008 ± 0.00006 (n.s.).
**Not adopted.** A second GBDT family adds nothing once the bge CE and the stacker sit on top. AWS ≈ $2.
