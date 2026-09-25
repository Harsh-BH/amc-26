# MERGE-1 — 25 Sep 2026, 11:20 IST (Wave A: P1, P2, P3, P7)

| Probe | Recommendation | Evidence (ledger rows) | Decision | Promoted to |
|---|---|---|---|---|
| P2-script | `translit.to_latin` on every S2/S3 name + address before normalising; `fold` for name n-grams; `oov_flag` as a feature | P2-script/name-dict+fold (dev r@5 0.996, AUC 0.999); master holdout re-check 95.2% exact folded hit on 54,843 pairs | **adopt** | `src/translit.py`, `src/pairs.py` |
| P1-taxonomy | `normalize.py` as the shared normaliser; rule D as the high-precision first stage; score everything dense | P1-taxonomy/exact-keys-D…-dense-generic 0.7831; lookalike-chains-v1 (50% of S1 share a core) | **adopt** | `src/normalize.py` |
| master | rules = P2 → P1 → rule D | pipe s2-rules holdout-dense **0.8201**, pair precision 0.9937 | **adopt** as stage 1 + submission #2 | `src/rules.py` (+ `test_rules.py`) |
| P3-block | SX→S1 reverse TF-IDF (name char-3 + address words, 50/50, hashed, state-partitioned); keep_r=2 (10 if no address), keep_f=10 | full-v4-{India,US}-derived_kr2_kf10_knoaddr10: pair recall **0.9916 at 14.2/S1** on holdout S1 vs the full 10.3M corpus / 2.2M S1 pool | **adopt** | `probes/P3-block/block.py` used in place until Phase 2; test candidates built: `data/work/p3/cands_test_*.parquet` (27.4M, 15.8/S1) |
| P7-scale | full run fits locally; per-country run shape; `chunks.write_lists`; P3 pool-pieces speed-up (exact) | P7 ledger rows; features 384k pairs/s, LightGBM 100–380k rows/s | **adopt run shape**; speed-ups only if blocking time binds | Phase 2 `run.py` |

## Conflicts and how they were settled
- **Plain vs dense evaluation.** P1: the plain holdout world hides chain collisions; P3: dev overstates full-corpus recall by ~2 pts.
  → Every number from now on is dense / full-density: holdout S1 scored while all train S1 and all train S2/S3 compete.
- **place_freq on test S1** (P1 wants it for France city keys) vs the plan's "fit on train only until Q4 is answered".
  → Train only for now. France falls back to a consistent city choice plus place overlap. Revisit when the Form answers.
- **P7 died** (OOM ~08:23, no DONE). Its partial RESULT already answers the scale question, and P3 measured the real test blocking run (35 min). Not resumed.

## Current best pipeline config
`s2-rules`: to_latin → name_core / address_fields (freq = train S1) → exact (country, core) or website key → address
score 4·hn + 2·city + place + state ≥ 2 → unique best S1 → each S2/S3 to at most one S1. Dense holdout F0.5 0.8201.

## Next wave (B)
- **P4-match.** Train a LightGBM matcher on full-density candidates.
  - Train on folds 2–9, exported from P3's full.py run. Evaluate on `cands_holdout_full_*`.
  - Candidates = P3 ∪ rule-key pairs.
  - Features: rules flags/score, block features, P1/P2 name and address similarities.
  - Target: beat 0.8201 dense by a clear margin.
- **P6-france.** French rules, leave-one-country-out with P4's features, Splink EM as an unseen-country check, and unsupervised France diagnostics on test.
- **Wave C after MERGE-2:** P5-decide on P4's holdout probabilities; P4b-rerank if a GPU slot and a clear uncertain band exist.
