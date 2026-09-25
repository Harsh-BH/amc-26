# P8-pipeline — RESULT
## TL;DR
DRAFT (test run in progress). `src/run.py` is a pluggable raw-TSV → both-files pipeline (feats → P3 ∪ rule keys →
rules | model.score → decide → per-country write + asserts + official validator); `test_pipeline.py` passes.
**Finding:** at true full density (all 10.3M train S2/S3 competing) the rules stage scores **0.7708**, not 0.8201 —
precision 0.912, singleton accuracy 0.710. The holdout world samples only ~5% of the look-alike distractors.

## Recommendation
Pending the test run.

## Evidence
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| P8-pipeline/src-block-equiv-dev | dev | src/block.py == probes/P3-block/block.py: 511,521 pairs identical, pair recall 0.9990 | keep |
| pipe p8-rules | holdout-dense (full corpus) | F0.5 **0.7708** (India 0.7893, US 0.7585), precision 0.9121, recall 0.6392, singleton 0.7103, cand recall 0.9917 at 14.2/S1; 21 min cold, 2.14 GB | keep (honest number) |
| pipe p8-rules | holdout-world-dense | F0.5 **0.8215**, precision 0.9940 — identical to test_rules_cands (by_state); 105 s | reproduces |
| P8-pipeline/rules-fullcorpus-vs-world-sx | holdout | 92.5% of full-corpus FPs are parentless S2/S3; 87.5% score exactly addr 4; 29% of singletons get an FP | — |
| P8-pipeline/rules-need-sweep-dense-fullcorpus | holdout | need 2 → 0.7708, need 4 → 0.7556, need 5 → 0.6603 (precision 0.9895): no threshold fixes it | drop |
| pipe p8-rules / p8-model | dev | rules 0.8526; stub LR 0.9802 (plain world, in-sample) | interface check |
