# P5-decide: from pair probabilities to answer sets
Inputs: data/work/P4/ out-of-fold probabilities (dev, holdout) + truth.
Compare (research_techniques.md §4): D0 global threshold · D1 exclusivity first (each SX only to its argmax S1;
many-to-one, so argmax is exact — NOT Hungarian) · D2 expected-F0.5-optimal subset per S1 (Ye et al. 2012;
exact DP of Jansche / Dembczynski) on isotonic-calibrated probabilities · D3 singleton gate (empty if max p < t0)
· D4 per-country thresholds with an unseen-country default · D5 top-m cap per S1 · D6 sibling expansion
(add SX whose near-duplicate S2/S3 sibling was matched) — test, don't assume.
Tune on dev, report on holdout, per country and singleton accuracy.
Deliverable: probes/P5-decide/decide.py: decide(pairs: DataFrame[s1, sx, p], cfg) -> dict[s1, set], plus the
winning cfg.

## UPDATE — Wave C runs IN PARALLEL with P4 (read probes/MERGE-1.md first)
- Don't wait for P4. Start now on data/work/p3/cands_holdout_full_{India,US}.parquet: full-density holdout candidates
  with is_true, 3.1M pairs. Add rule features from code/business_entity_resolution/src/rules.py.
  Build a quick proxy probability — e.g. logistic regression on block_score, block ranks, via, rule flags — with 2-fold
  cross-fitting over holdout S1. Develop and benchmark your decision strategies on it.
  Dense scoring: truth = worlds.load_world("holdout") truth. A true pair missing from the candidates is a recall loss.
- Poll data/work/p4/prob_holdout.parquet. When it appears, rerun the benchmarks on P4's probabilities and base the final
  recommendation on those.
- Tune on half the holdout S1 (hash split); report on the other half.
- Exclusivity is many-to-one: each SX goes only to its argmax S1. Do NOT use Hungarian (it would cap S1s at one match).
- Beat the rules baseline, dense 0.8201.
- Interface unchanged: `decide(pairs[s1, sx, p], cfg) -> dict[s1, set]`.
