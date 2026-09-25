# P9-integrate — RESULT   (DRAFT, test / holdout runs in progress)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- In progress. `run.py --stage model` now runs P3 ∪ rule pairs → M2 → CE blend (India/US) → argmax + per-country t, all from `src/`.
- Done so far: the promoted feature builder rebuilds P4's cached X files **exactly** (0 differing cells on test France and
  train India file 000). The dev world runs the whole in-memory path at F0.5 0.9949 in 100 s.
- P6's rule gives sub4's France t = 0.82 only with the reference on **M2's** US+India p at M2's t* 0.71; with blended p it gives 0.83.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
pending the test / holdout runs

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| P9-integrate/x-rebuild-equiv-test-France | test | France_sx_000: 1,526,605 rows × 67 cols, **0 cells differ** from P4's cached X; 49 s, 2.5 GB | keep |
| P9-integrate/x-rebuild-equiv-train-India | train | India_sx_000: 1,550,123 rows (incl. is_true, fold), **0 cells differ**; 79 s, 2.7 GB | keep |
| pipe p9-model | dev | F0.5 0.9949 (India 0.9958, US 0.9943), P 0.9986, R 0.987, singleton 0.9965; 100 s, 0.81 GB | keep (plain world: interface + recompute check) |
| (check, no row) | test | P6 rule on cached probs: ref = M2 p @ 0.71 → t_France 0.82; M2 @ 0.70 → 0.82; blend @ 0.70 → 0.83 | reference = M2 p |

## What failed      (approaches tried and dropped, with the number that killed them)
pending

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
All in `code/business_entity_resolution/src/` (branch wave/p9-integrate):
- `features.py`: P4 features.py + featrun.py.
  - `rec(df)`, `featurize(pairs, r1, rx, i, j, rule_acc)` in memory; it mirrors the files exactly (3-decimal rounding, NaN
    context for rule-only pairs).
  - `build(split, c)` (ctx_step + feat_step) → `data/work/p4/X/<split>_<c>_sx_NNN.parquet`, resumable per SX file.
- `model.py`: M2 behind P8's plug point.
  - `score(pairs, f1, fx)` in memory; `split_probs(split, c) -> [q1, qx, p]` from the X files.
  - Cached at `data/work/p9/m2_<split>_<c>_<booster sha1[:8]>.parquet`. `split="holdout"` keeps every pair of the SX that
    fold-0 S1 compete for (11.5M of 31.4M train pairs).
- `train_matcher.py`: `main(out)` fits fold 2 and early-stops on a 25% fold-1 hash sample (P4's ladder, same row order);
  `holdout([boosters])` gives the M2-only dense F0.5.
- `rerank.py`: P4b minus the analysis functions. `band`, `train` (+ `n=` smoke option), `score_file` (the pipeline step:
  [ta, tb] → [r]), `blend`, `fit_blend`, `texts`.
- `run.py`:
  - `model_parts`, `rerank` (subprocess under `flock /tmp/amc-gpu.lock`, r cached per split / country / CE hash),
    `thresholds` (P6 rule), `refit_w`.
  - `T = 0.70`, `T_M2 = 0.71`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- France t sits on a flat, close boundary: France's empty rate is 5.682% at 0.82 against the 5.691% reference, and 5.737% at 0.83.

## Open questions   (for the master; each with your best guess)
pending
