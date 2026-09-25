# P14-legal — RESULT
## TL;DR
Canonicalising legal forms ("Private Limited" = "Pvt. Ltd." = "Pirvate Limted" → `pvt ltd`) and adding 7 label-free
features raises the full-corpus holdout F0.5 of the fold-2 M2 from **0.9814 to 0.9838 (+0.0024 ± 0.0001, paired SE,
half 1; 3,105 S1 up, 945 down)**. India +0.0032, US +0.0019. Recall and precision both improve.
Measured **without** the cross-encoder: confirm on top of the #5 recipe (folds 2–5 + CE blend) before promoting.

## Recommendation
- Next: the same A/B with the cross-encoder blend and on folds 2–5. The bar is still +0.001. More training folds
  gained +0.0019 without the CE but only +0.0004 with it (P4 improvement round), because the CE fixes the same
  uncertain band. The CE also reads the legal words, so expect some shrinkage here too.
- If it holds, promote `canon` / `extras` into `src/features.py` as 7 columns after the 65. Then rebuild the X files
  (train + test) and retrain M2. Refit the blend weights, T and T_M2. Recompute France's threshold with P6's rule.
- France is only measurable on the leaderboard: 69% of French S1 names carry SARL/SAS/…, and those forms are in
  `N.LEGAL`.

## Evidence
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| P14-legal/M2-fold2-base | holdout-full (half 1) | F0.5 **0.9814** (India 0.9803, US 0.9821), t 0.73; = P4's M2-lightgbm 0.9814 | control |
| P14-legal/M2-fold2-legal | holdout-full (half 1) | F0.5 **0.9838** (India 0.9835, US 0.9840), t 0.70; val logloss 0.0256 → 0.0224, 2,222 → 1,697 trees | keep (≥ 0.001) |

**Same rows, same folds, same params.** The only difference between the two models is the 7 extra columns:
- Training: M2 recipe (`model.PARAMS`), fitted on fold 2 (3,142,744 pairs), early-stopped on the fold-1 25% sample
  (793,402 pairs).
- Scoring: like `run.py --split holdout`. Every pair of each S2/S3 that a fold-0 S1 competes for is scored; each
  S2/S3 goes to its argmax over all train S1; one threshold is tuned on half 0; F0.5 is reported on half 1.

| holdout, argmax pairs at the tuned t | base | legal |
|---|---|---|
| true matches kept | 731,539 | **735,772** (+4,233) |
| wrong matches kept | 4,114 | **3,743** (−9%) |
| … of which the two names' legal forms conflict | 709 | **145** |

**Diagnostics** (fold-0 true candidate pairs):
- Of the true pairs where both names carry a legal form, 91.4% have the same canonical form and 8.6% differ
  (363,041 vs 33,993). The generator does swap legal types sometimes, so a conflict is evidence, not a rule.
- Legal-form prevalence: India 84% of S1 / 78% of S2-S3 names; US 57% / 50%.
- Feature gain rank among 72: `legal_jacc` 8, `cname_ratio` 10, `legal_n1` 14, `legal_nx` 16, `cname_tset` 22.
  The 7 new columns hold 4.1% of total gain.

## What failed
Nothing was tried and dropped. Not run:
- the per-feature ablation;
- the cross-encoder blend on top;
- folds 2–5;
- more than one training seed per arm. P9 measured retrain noise below 0.0001, against the observed +0.0024.

Engineering:
- `features.ctx_step` holds a country's whole candidate table and is OOM-killed on US train (18.6M pairs, no swap,
  4.5 GB cap). `build.py` replaces it with `ctx_lean`, which is verified identical on all 8 India files (every row
  and value).
- Scoring whole X files with Python-string names also OOM'd. `legal.py` keeps the names as Arrow strings and scores
  in 400k-row chunks.

## Interfaces
- `legal.py canon(names) -> (bitmask, canonical name)`: 320k names/s. Its input is P3's cleaned `name`.
- `legal.py recs(c)`: per-record masks and canonical names in q1 / qx order, cached at `data/work/p14/rec_*`.
- `legal.py extras(q1, qx, recs) -> float32 [n, 7]`, rounded to 3 decimals like the cached features.
- `legal.py run`: ~19 min once the X files exist, peak 4.1 GB. Writes `data/work/p14/{m2_base,m2_legal}.txt` and
  `ab.json`.
- `build.py`: the train side via the pipeline's own functions plus `ctx_lean`. The cleaned names came from Harsh's
  Drive, and their row order was checked against this machine's parquet. India took 36 min, US 7 min after its
  blocking. 31,404,829 train pairs.

## Risks
- **The CE overlap** above: the gain may shrink after the blend.
- **Over-stripping.** `N._is_legal`'s typo match also catches real words, e.g. "Corporate" → `corp` and "Primate" →
  `pvt`. The effect is symmetric between records and has little gain, but a tighter matcher could do better.
- **France.** French forms (`sarl`, `sas`, …) are never seen in training. `legal_eq` / `legal_diff` / `legal_jacc`
  are country-agnostic, so their meaning should transfer, but that is unverified.
- **Licences / external data:** none. The features are pure string functions of the records.

## Open questions
1. Does the gain survive the CE blend and folds 2–5? Best guess: yes, but smaller, around +0.001.
2. Which half carries it: canonical-name similarity (recall) or the conflict features (precision)? The +4.2k true
   matches suggest the spelling unification carries more.
