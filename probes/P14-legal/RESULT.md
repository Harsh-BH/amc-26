# P14-legal — RESULT
## TL;DR
In the full #7 pipeline (pruner + m2 on folds 2-5 + 2-epoch CE), the 7 legal-form features move the full-corpus
holdout F0.5 from **0.98867 to 0.98880 (+0.00013 ± 0.00006, half 1)**, below the +0.0005 bar. The cross-encoder
already fixes most of what they fix. Without the CE the gain is large and stable: **+0.00175** on folds 2-5 and
+0.0024 on fold 2. France is scored without a CE, so a leaderboard upload is the only way to see its effect.
`output/sub_legal/` holds the test files (recipe `sub7L`).

## Recommendation
- India / US: no measurable gain through the CE (India +0.0003, US +0.00003). Keeping the features costs nothing,
  but they do not justify a submission on their own.
- France is the open question. It is where the LB gap is, it has 69% legal forms in S1 names, and it is scored by
  M2 alone, the regime where the features gain +0.0018 on the labelled countries. `output/sub_legal/` =
  `run.py --split test --stage model --recipe sub7L`. France t is 0.96, the threshold giving #7-France's empty-S1
  rate: 6.80% vs 6.85%. Upload it if the team wants the France answer (Harsh's call). Compare it with #7's LB score.

## Full pipeline (holdout, full corpus; tuning on half 0, F0.5 on half 1) — `full.py`, pipe ledger p14-*-sub7
| arm | w (refit) | t | F0.5 half 1 | India | US | all fold 0 |
|---|---|---|---|---|---|---|
| #7 as shipped (w, t of the recipe) | [0.617, 0.666, 0.261] | 0.71 | 0.98869 | | | **0.98881** (= #7's number) |
| base: m2_big | [0.6236, 0.6702, 0.2674] | 0.63 | **0.98867** | 0.98964 | 0.98802 | 0.98883 |
| legal: m2_legal_big | [0.6239, 0.641, 0.2264] | 0.65 | **0.98880** | 0.98993 | 0.98806 | 0.98895 |
| base, M2 alone (no CE) | — | 0.71 | 0.98329 | 0.98274 | 0.98366 | 0.98334 |
| legal, M2 alone (no CE) | — | 0.72 | 0.98504 | 0.98512 | 0.98499 | 0.98506 |

legal − base on half 1: **+0.00013 ± 0.00006** (paired SE; 367 S1 up, 364 down, n = 110,362).
m2_legal_big: 3,571 trees, fold-1 val logloss 0.02049 (m2_big: 4,930 trees, 0.02285), fit in 20 min at a 4.35 GB peak.

## Fold-2 experiment (no CE) — `legal.py`
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
