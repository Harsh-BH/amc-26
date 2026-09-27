# Wake-up brief — 27 Sep (updated as results land)

## Upload this first
**`output/sub_iu54_frgc70/`** — validator PASS. Upload matching_results.tsv, candidate_pairs.tsv, code.zip.
= sub54's India/US (new combiner) + gc70's France (LB-proven). Expected **≈ 0.9867–0.9869** (≥ sub54's 0.986685).

Why: sub54 (0.986685) got only +0.00038 over gc70. Two independent checks (P57 honest frame, P62 test-density clones)
show sub54's India/US combiner alone is worth +0.0004–0.0006, so sub54's extra France rules netted ≈ −0.0002 to 0.
This file keeps the proven India/US gain and drops the unproven France changes.

## Upload order for the remaining slots (round closes 27 Sep 23:59 IST)
1. `sub_iu54_frgc70` — the safe best (above).
2. `sub_clean` — gc70-era France rules, cleaned (no P46 "Développement", no P40 no-address names) + P55 combiner, with the same
   India/US. If it beats #1, the cleaned French rules help; if not, gc70's France stays.
3. Variants P50 is staging on #1 (see SUBMISSIONS.md): `sub_iu54_fr38` (France reranker swapped to Qwen, +0.0002 expected),
   `sub_iu54_fr55b` (P55 veto only), and P61 co-training / P59–P60 bigger-data rerankers if they pass.
4. Keep one slot spare.

## LB history
D4 0.981142 → sub9ab 0.984072 → sub31 0.9856 → sub34_veto 0.985647 → rescue90 0.98596 → gc70 0.986304 → **sub54 0.986685**.
Top 3 now: 0.991483 / 0.991043 / 0.990881.

## What worked / what didn't (27 Sep)
- Worked: India/US learned combiner + bge scored on every pair (+0.0005 on honest/test-density checks).
- Didn't transfer: most France rules added after gc70 (net ≈ 0 on LB); ensembles, twin tricks, shape matching, serialization,
  finer thresholds (all ≈ 0 on holdout).
- Public repos: no top-30 team repo is public. Useful lessons from others: France is precision-limited; density shift.
