# P9-integrate: one command that reproduces submission #4 from the raw data (the final package depends on it)
You run in your own worktree (branch wave/p9-integrate) and MAY edit code/business_entity_resolution/ there. Commit on
your branch as you go (never push, never touch main). Read probes/MERGE-2.md, probes/P8-pipeline/RESULT.md,
probes/P4b-rerank/RESULT.md, probes/P6-france/RESULT.md first. P4's code snapshot is in probes/P4-match/ (copied from
its worktree; P4 is still running in parallel on an improvement — do not depend on files it may still change under
data/work/p4/ other than the ones named below).

Target = submission #4, staged at output/sub4/ in the main checkout (/home/harsh1/github-repos/amc-26/output/sub4/):
P3 blocking ∪ rule pairs → P4 M2 LightGBM (data/work/p4/m2_ladder.txt, 65 features) → P4b cross-encoder blend on
0.02<p<0.98 for India/US (data/work/p4b/ce.pt, weights w=[0.636,0.757,0.355]) → argmax per S2/S3 → p ≥ t
(India/US 0.70 on blended p; France 0.82 on M2 p, from P6's label-free rule fr_threshold).

Build (in src/, keep it small, reuse src/run.py's plug points and P4/P4b/P6 code rather than rewriting):
1. `features.py` — P4's 65-feature builder (promote from probes/P4-match; same column order as the booster expects).
2. `train_matcher.py` — trains M2 from the TRAIN data exactly as P4 did (fold 2 train, fold 1 early stop), writes the
   booster. Must reproduce m2_ladder.txt's holdout number within ±0.001 (you may validate on a sample if a full retrain
   is too slow — say which).
3. `rerank.py` — promote P4b's rerank.py (fix its ROOT/path assumptions for src/); include its training entry point.
4. `run.py --stage model` — wire features → booster → rerank (India/US, own process under flock /tmp/amc-gpu.lock) →
   decide (argmax + per-country t; France t computed by P6's rule from the test probabilities, not hard-coded) → both
   TSVs → asserts → official validator.
5. README.md — the exact end-to-end reproduction: data → blocking → training (M2, CE) → inference → output, with
   runtimes and memory, and which artefacts are cached where.
Acceptance:
- `run.py --split test --stage model` reproduces output/sub4/matching_results.tsv: identical lines on >= 99.99% of S1
  (report the diff count and why any lines differ). Reuse cached intermediates (data/work/p3, p4, p4b, rf_test_*) where
  they exist, but the code path that recomputes each must exist and be tested on the dev world.
- `run.py --split holdout --stage model` = full-corpus holdout 0.988 ± 0.001; append a `pipe` ledger row.
- test_pipeline.py passes; package.py builds the zip and its structure check passes.
Memory cap 3 GB incl. the CLI, swap allowed. GPU only under the flock. Disk ~13 GB free: do not duplicate big caches.
