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
