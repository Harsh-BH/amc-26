# P25-lgbm-bag — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- INTERIM (15:30). Eight of nine LightGBM variants are trained and evaluated on the #8c yardstick: bag1–4, GOSS, deep, shallow, extra_trees. DART is training (3,000 rounds, done ~16:40).
- No config clears the bar (+0.0003 at 3 SE). Equal-weight bag of all 8 + m2_big: **+0.000001 ± 0.000040**. Fitted blend (LR on logits, half 0): +0.000015 ± 0.000048; non-negative fit +0.000024 ± 0.000049. Best single config: m2_big + bag4 at +0.000063 ± 0.000043 (1.5 SE).
- Leaning **drop**. Shallow, extra_trees and DART are still training; the final eval runs when they finish.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Pending the final nine-model eval. Interim: nothing; keep m2_big.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| rehearse-1pct-dart-goss | train-1pct-slice | solution_p25.py on AWS (ml.m5.xlarge): 1% S1 of folds 2–5; DART no-early-stop path and GOSS params valid; all 2,783,306 holdout and 8,370,573 test keys scored; 70 s | keep |
| sm-jobs-launch | aws | 9 jobs `p25-<variant>-0926-1404`: 4× m6i.4xlarge, 2× c7i.4xlarge, 2× r6i.4xlarge, 1× m7i.4xlarge; MaxRuntime 3 h, training deadline 8,400 s | — |
| M2-{bag1..4,goss,deep}-train | train | fold-1 val logloss (m2_big 0.02285): bag1 0.02303 (4,962 trees), bag2 0.02311 (4,331), bag3 0.02304 (4,832, folds 6–9), bag4 0.02306 (4,521, folds 6–9), goss 0.02309 (3,303), **deep 0.02278** (2,861). 18–27 min each, peak ≤ 5.8 GB, $0.33–0.43 each | keep (pair level) |
| (eval repro) | holdout-dense | the AWS eval job reproduces #8c's pipe row exactly: 0.98990 / 0.99024 / 0.98956 | — |
| eval-six-8c | holdout-dense-half1 | table below. Best: avg_bag4 +6.3 ± 4.3 (1.5 SE), avg_bag +5.5 ± 4.6, lr_all +2.3 ± 4.7, nn_all +1.4 ± 4.8 (units of 1e-5) | drop |

Eight-model eval (p25-ev-0926-151340): half 1 of the holdout, per-S1 F0.5, paired vs #8c as shipped (0.98956). Gains are in **1e-5 F0.5**, with gain/SE in brackets. Main column: w and the US stacker refit on half 0, t 0.71.
- "vs ctl" = the control (m2_big) put through the same refit procedure.
- "band w/o bge" = band pairs of holdout S1 with no bge r; they take the e5 fallback.

| config | half-1 F0.5 | vs #8c (refit stk, t .71) [SE] | India | US | vs ctl same procedure | vs #8c ship stk | vs #8c t swept | band w/o bge |
|---|---|---|---|---|---|---|---|---|
| alone_bag1 | 0.98952 | -4.8 ± 4.6 [-1.1] | -0.7 | -7.6 | -3.5 ± 4.6 [-0.8] | -6.1 ± 4.4 [-1.4] | -5.2 ± 4.7 [-1.1] | 10710 |
| alone_bag2 | 0.98949 | -7.7 ± 4.7 [-1.6] | -13.8 | -3.7 | -6.4 ± 4.8 [-1.3] | -7 ± 4.5 [-1.5] | -8.9 ± 4.7 [-1.9] | 12855 |
| alone_bag3 | 0.9895 | -6.3 ± 5.8 [-1.1] | -9.8 | -4 | -5 ± 5.7 [-0.9] | -3.8 ± 5.5 [-0.7] | -15 ± 6.4 [-2.3] | 16352 |
| alone_bag4 | 0.98959 | 2.4 ± 6.1 [0.4] | -1.1 | 4.7 | 3.7 ± 6.1 [0.6] | 1.8 ± 5.7 [0.3] | -2.9 ± 6.1 [-0.5] | 17324 |
| alone_goss | 0.98949 | -7.1 ± 5.3 [-1.3] | -11.9 | -3.9 | -5.7 ± 5.1 [-1.1] | 1.6 ± 4.8 [0.3] | -9.6 ± 5.4 [-1.8] | 16868 |
| alone_deep | 0.98951 | -5.8 ± 5.3 [-1.1] | -5.7 | -5.9 | -4.5 ± 5.1 [-0.9] | -2 ± 4.6 [-0.4] | -7.3 ± 5.1 [-1.4] | 8251 |
| alone_shallow | 0.98958 | 1.7 ± 4.6 [0.4] | 2.1 | 1.4 | 3 ± 4.8 [0.6] | 2.8 ± 4.4 [0.6] | -2.5 ± 4.8 [-0.5] | 16837 |
| alone_extra | 0.9894 | -16.6 ± 6.3 [-2.6] | -17.2 | -16.1 | -15.2 ± 6.4 [-2.4] | -20.6 ± 6.3 [-3.3] | -15.3 ± 6.3 [-2.4] | 20300 |
| avg_bag1 | 0.98954 | -2.4 ± 2.8 [-0.8] | 0.2 | -4 | -1 ± 3.1 [-0.3] | -4.6 ± 3.1 [-1.5] | -2.4 ± 2.8 [-0.8] | 5136 |
| avg_bag2 | 0.98955 | -1.7 ± 3.3 [-0.5] | -1.8 | -1.7 | -0.4 ± 3.4 [-0.1] | -0.2 ± 3.1 [-0.1] | -0.2 ± 3.5 [-0.1] | 6088 |
| avg_bag3 | 0.98957 | 0.9 ± 4.3 [0.2] | 2.2 | 0.1 | 2.3 ± 4.1 [0.5] | 4.4 ± 4.3 [1] | -5.6 ± 5 [-1.1] | 8052 |
| avg_bag4 | 0.98963 | 6.3 ± 4.3 [1.5] | 5 | 7.3 | 7.7 ± 4.2 [1.9] | 6.5 ± 4.4 [1.5] | 3.2 ± 4.9 [0.6] | 8370 |
| avg_goss | 0.98956 | -0.3 ± 3.4 [-0.1] | -1.5 | 0.5 | 1 ± 3.6 [0.3] | 1.4 ± 3.7 [0.4] | -0.6 ± 3.7 [-0.2] | 7993 |
| avg_deep | 0.98952 | -4.8 ± 3.8 [-1.3] | -8.5 | -2.4 | -3.5 ± 4 [-0.9] | -5.2 ± 3.5 [-1.5] | -6.4 ± 4.1 [-1.5] | 4045 |
| avg_shallow | 0.9896 | 4.1 ± 3.7 [1.1] | 2.3 | 5.3 | 5.4 ± 4 [1.4] | 2.8 ± 2.8 [1] | 1.4 ± 3.7 [0.4] | 8115 |
| avg_extra | 0.98947 | -9.5 ± 4.5 [-2.1] | -8.3 | -10.3 | -8.2 ± 4.7 [-1.7] | -4 ± 3.7 [-1.1] | -8 ± 4.6 [-1.7] | 9554 |
| avg_bag | 0.98962 | 5.5 ± 4.6 [1.2] | 4.9 | 5.9 | 6.9 ± 4.4 [1.5] | 6.4 ± 4.3 [1.5] | 5.5 ± 4.6 [1.2] | 8776 |
| avg_boost | 0.98957 | 0.2 ± 4 [0] | -0.9 | 0.9 | 1.5 ± 4.2 [0.4] | 3 ± 4 [0.7] | -4.9 ± 4.3 [-1.1] | 9017 |
| avg_data | 0.98959 | 2.8 ± 5 [0.6] | 5.2 | 1.2 | 4.2 ± 5 [0.8] | 8.5 ± 4.8 [1.8] | -2.1 ± 5.2 [-0.4] | 10477 |
| avg_all | 0.98956 | 0.1 ± 4 [0] | 2.5 | -1.5 | 1.5 ± 4.2 [0.4] | 1.6 ± 4.2 [0.4] | -0.1 ± 4.7 [-0] | 9037 |
| avg_var | 0.98958 | 2 ± 4.3 [0.5] | 4.9 | 0 | 3.3 ± 4.5 [0.7] | 4.8 ± 4.2 [1.2] | 2 ± 4.3 [0.5] | 10192 |
| lr_all | 0.98958 | 1.5 ± 4.8 [0.3] | 0.3 | 2.3 | 2.8 ± 4.7 [0.6] | 0.7 ± 4.5 [0.1] | 1.5 ± 4.8 [0.3] | 5203 |
| nn_all | 0.98959 | 2.4 ± 4.9 [0.5] | -1 | 4.6 | 3.7 ± 4.8 [0.8] | 2.5 ± 4.4 [0.6] | 0.6 ± 4.8 [0.1] | 5109 |

Pair level (half-1 holdout S1's kept pairs):

| model | logloss India | logloss US | band logit corr with m2_big (India / US) |
|---|---|---|---|
| m2_big | 0.05756 | 0.06301 | 1 |
| bag1 / bag2 (folds 2–5) | 0.05801 / 0.05825 | 0.06345 / 0.06355 | 0.982 / 0.985 |
| bag3 / bag4 (folds 6–9) | 0.05773 / 0.05781 | 0.06330 / 0.06332 | **0.958 / 0.962** |
| goss | 0.05820 | 0.06351 | 0.982 / 0.985 |
| deep | **0.05742** | **0.06270** | 0.982 / 0.985 |

- Fitted weights (logit space, both countries pooled, half 0):
  - `nn_all`: m2_big 0.19, bag3 0.27, bag4 0.16, deep 0.41; bag1, bag2 and goss get 0.
  - `lr_all` has the same shape, with small negative weights on bag2 and goss.

## What failed      (approaches tried and dropped, with the number that killed them)
- Interim, six models: every single variant, every m2_big + variant pair, the equal-weight bags and both fitted blends. All are within ±2 SE of #8c; none is above +0.00007.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `solution_p25.py`: the SageMaker training job, one LightGBM variant per job. It is P16's training path plus P22's key scorer.
  - Writes `<variant>.txt` (booster), `pred/{holdout,test}_<c>.parquet` (q1, qx, p; #6's pruned rows, France included), `status.json` and `result.json`.
  - Destination: `s3://sagemaker-114872709824-ap-south-1/amc26-p25/results/<job>/`.
  - Full scale: 18–27 min on a 16-vCPU instance, ≤ 5.8 GB.
- `variants.json`: variant = m2_big's params + overrides, plus folds, rounds and instance type.
- `launch.sh v1 v2 …`: submits the jobs (bash + aws cli + jq; no Python). `status.sh`: one status line per job.
- `ev25.py` + `solution_p25ev.py` + `ev_launch.sh '<runs>' v1 v2 …`: the #8c-yardstick evaluation on AWS.
  - It imports P22's `ev.py` unchanged. The channel is P22's `amc26-p22/data/ev/`.
  - ~8 min on ml.m5.4xlarge, 1.6 GB.
- `report.sh eval.json`: turns an eval JSON into the markdown table above.
- `ledger_row.sh`: writes ledger rows without Python (the laptop's memguard pauses probe Pythons).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- LightGBM 4.7.0 is MIT. No new dependencies.
- Band pairs with no bge r fall back to the e5 blend, as `CE_MISSING=fallback` does in #8c (4k–17k holdout-S1 pairs per config). This slightly handicaps the variants: production would bge-score them. It cannot explain a gap to +0.0003.
- About 25 configs are compared on the same half 1. The best at 1.5 SE is what the maximum of 25 noisy draws looks like.

## Open questions   (for the master; each with your best guess)
- Pending.

## Master conclusion (16:02) — stopped, not adopted
Eight diverse LightGBM variants (4 bags, GOSS, deep, shallow, extra_trees) on the #8c yardstick: equal-weight bag of all
8 + m2_big +0.000001 ± 0.000040, fitted blend +0.000015 ± 0.000048, no single variant above noise. With P22's result
(XGBoost/CatBoost ≈ 0), a second tree model adds nothing once the bge CE + P17 stacker sit on top. DART (still training)
was stopped by the master to save cost; not evaluated. ≈ $6 AWS.
