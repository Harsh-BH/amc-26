Master decision (MERGE-2): the matcher now includes P4b's cross-encoder blend for India/US. The blended holdout
probabilities are in data/work/p4b/prob_holdout_blend.parquet (P4's prob_holdout schema, p = blended p', plus p_gbdt).
Refit your D7 stack on the blended p exactly as your RESULT.md says (bench_stack.py ... with SRC pointing to that file):
1. Report, on the same half-split protocol, F0.5 for (a) FALLBACK threshold on blended p (P4b found t 0.70) and
   (b) your full CFG (stack + excl + EF top-k + gate) refit on blended p. Paired SE as before.
2. Write the refit stack/cfg under new names (stack_blend_s1only.txt, cfg_blend_s1only.json) — do NOT overwrite the
   M2 ones. Log ledger rows. Update RESULT.md with a short "Blend refit" section and the recommendation:
   adopt the stack on blended p only if (b) beats (a) by >= 0.001.
3. If adopted, also produce the test decision inputs: note that test blended probs will appear at
   data/work/p4b/prob_test_{India,US}_blend.parquet (France stays on data/work/p4/prob_test_France.parquet, and France
   uses P6's threshold 0.82 — not your stack). Then stop.
Cap 2 GB incl. the CLI (swap allowed).
