# Wake-up brief — 27 Sep (updated as results land)

## Upload this first
**`output/sub_iu54_frgc70/`** — validator PASS. Upload matching_results.tsv, candidate_pairs.tsv, code.zip.
= sub54's India/US (new combiner) + gc70's France (LB-proven). Expected **≈ 0.9867–0.9869** (≥ sub54's 0.986685).

Why: sub54 (0.986685) got only +0.00038 over gc70. Two independent checks (P57 honest frame, P62 test-density clones)
show sub54's India/US combiner alone is worth +0.0004–0.0006, so sub54's extra France rules netted ≈ −0.0002 to 0.
This file keeps the proven India/US gain and drops the unproven France changes.

## Upload order for the remaining slots (round closes 27 Sep 23:59 IST)
1. **`sub_iu54_frgc70`** (safe best): sub54 India/US + gc70 France. Expected ≈ 0.9867–0.9869.
2. **`sub_iu54_best`**: #1 + France extras that all passed checks: Qwen reranker for France (LOCO +0.0011–0.0014 France-F,
   8 SE both ways), P55 veto that leaves LB-proven rescues alone (−619), acronym rule from public-repo ideas (+3,538; 99.7–99.97%
   precise on train), dup consistency. Expected ≈ 0.9870–0.9873. If #2 > #1 → #2 is the final base.
3. `sub_clean` (old France rules cleaned + P55) — only if you want to test whether post-gc70 French rules can help at all.
4. Keep one slot spare for the final combined file (+ P59/P60 bigger-data rerankers if they pass).
Fallbacks staged: sub_iu54_fr38 (Qwen only), sub_iu54_frgc70_acro (acronym only), sub_iu54_fr55b_norescue, sub61h9.

## LB history
D4 0.981142 → sub9ab 0.984072 → sub31 0.9856 → sub34_veto 0.985647 → rescue90 0.98596 → gc70 0.986304 → **sub54 0.986685**.
Top 3 now: 0.991483 / 0.991043 / 0.990881.

## What worked / what didn't (27 Sep)
- Worked: India/US learned combiner + bge scored on every pair (+0.0005 on honest/test-density checks).
- Didn't transfer: most France rules added after gc70 (net ≈ 0 on LB); ensembles, twin tricks, shape matching, serialization,
  finer thresholds (all ≈ 0 on holdout).
- Public repos: no top-30 team repo is public. Useful lessons from others: France is precision-limited; density shift.

## Final zip (ready)
`output/final/Yeti_submission.zip` (93.8 MB, 30 files, organisers' layout) = `sub_iu54_frgc70` with the SMALL candidate set
(4.42 candidates per S1, ranks higher beyond the LB). Official validator PASS on the folder and on the unzipped zip.
`run.py` (default recipe `sub_iu54_frgc70`) rebuilds the matching file byte for byte. Docs filled; `[FINAL LB]` slot left for
the final score. Backed up to S3 (amc26-offload/final/) and Drive (gdrive:amc26-offload/final/). Code pushed to main.
If `sub_iu54_best` scores higher on the LB, P43 re-zips it (its stages are being ported now; ~30 min).
