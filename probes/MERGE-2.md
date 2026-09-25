# MERGE-2 — 25 Sep 2026, ~15:25 IST (Waves B + C: P4, P5, P6, P8, P4b)

Yardstick for every number: **full-corpus holdout**. Train fold-0 S1 are scored while all 2.2M train S1 and all 10.3M
train S2/S3 compete. The "dense world" used before 12:17 overstated precision (rules 0.8215 → 0.7708).

| Probe | Recommendation | Evidence (ledger rows) | Decision | Where |
|---|---|---|---|---|
| P8-pipeline | `src/run.py` end-to-end; `src/block.py` (P3 promoted); model/decide plug points; packaging | P8-pipeline/src-block-equiv-dev (identical to P3); run.py test output byte-identical to submission #2 | **merged** (f133388) | `src/` |
| P4-match | LightGBM on 65 label-free features over P3 blocking ∪ rule pairs; argmax per S2/S3, t 0.71 | P4-match/M2-lightgbm **0.9814** (P 0.994, R 0.959). Two independent Opus audits: clean, recomputed 0.981413 | **adopt**, submission #3 | `data/work/p4/`, code in `probes/P4-match` (to merge when P4 exits) |
| P5-decide | D7 stack + exclusivity + EF top-k behind a gate | P5-decide/p4-D7-stack-s1only+ef-gate **0.9824** (+0.0010 ± 0.0001); FALLBACK t 0.71 = 0.9815 | **adopt later**: the stack is tied to M2's p, so refit it on the CE-blended p before use | `probes/P5-decide/decide.py` |
| P4b-rerank | e5-small cross-encoder rescores the 0.02<p<0.98 band, India + US only | P4b-rerank/p4-D3-blend-0.02-0.98 **0.9881** (+0.0066 ± 0.0002); LOCO −0.0016, so no France | **adopt**, submission #4 | `probes/P4b-rerank/rerank.py`, `data/work/p4b/ce.pt` |
| P6-france | pooled US+India model; t_France from France's empty-S1 rate vs US+India test at t* | P6-france/loco-dense-threshold-policies: rule = oracle (0.9054) US→India; P6-france/france-threshold-prob_test → 0.82 | **adopt** | `probes/P6-france/fr_threshold.py` |

## Conflicts settled
- **Decision layer vs reranker.** Both change the final sets. P4b's gain (+0.0066) is 6× P5's (+0.0010), and P5's stack
  must be refit on whatever p it sees. Order: blend first (#4), then refit P5's stack on the blended holdout p and add it
  only if it gains ≥ 0.001 on the same yardstick (#5).
- **France threshold.** P5's D4 picks 0.71 for both labelled countries, but it has no France data. P6's LOCO shows an
  unseen country needs a stricter threshold, and its label-free rule matched the oracle. Use P6: 0.82. Submission #3
  shipped 0.83 (between P6's 0.82 and 0.84 estimates); both sit in the flat region.
- **Orphan density risk (audit).** 87% of holdout false positives are orphan S2/S3, and test has more S2/S3 per S1
  (5.8 vs 4.7). At t 0.71 the test outputs reproduce the holdout profile (3.36 matches/S1, 5.7% empty vs 3.35 / 5.65%),
  so the risk looks small. P6's red-flag bands guard France.

## Current best pipeline config
P3 blocking ∪ rule pairs → P4 M2 LightGBM → [India, US: P4b CE blend on 0.02<p<0.98, w = 0.636 / 0.757 / 0.355] →
argmax per S2/S3 → p ≥ t (India/US 0.70 on blended p; France 0.82 on M2 p).
Expected full-corpus holdout ≈ 0.988.

## Next
1. #4 = the config above (CE test scoring running, then submit_probs + validator).
2. Refit P5's stack on `data/work/p4b/prob_holdout_blend.parquet`, then #5 if +0.001.
3. P4 finishes → merge its code into src (model.py score interface) so run.py reproduces #3/#4 end to end.
4. Improvements queue: P4 retrained on folds 2–9 (it used fold 2 only), France normalisation rules (P6 france.py),
   CE for France via pooled fine-tuning is **not** planned (LOCO says it hurts).
