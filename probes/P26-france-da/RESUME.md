Master here — I interrupted you briefly to hand over a finding from P20 (probes/P20-france-patterns/RESULT.md, now on
main; `git merge main` to read it) that is directly relevant to your Phase A/B:
- France's biggest remaining gap is ~75k "one core word swapped, house number EQUAL" pairs: #8c accepts only 35% of them,
  while the same cell in the US is 89% true. Their logit is 4.8 below the US cell; **sx_margin explains −2.1 of it**
  (France 0.12 vs US 0.36): a competitor that the pruner later removes (93% of the time) — i.e. context features computed
  over the FULL blocking list, which is an IDF/blocking artefact on France. Shorter names explain about −2 more.
- France's generated siblings keep the hn only 2–6% of the time (like the US 2.7%, unlike India 26%), so hn-equal pairs
  should mostly be true in France.
- A blanket quantile-mapping of sx_margin (P20's R3) failed; P20 says it needs a model-level change: **context features
  over the PRUNED list**, or a France-invariant retrain. Please include "context features over the pruned list" as an
  arm in your Phase B (LOCO), alongside reweighting and hn features in the main model.
Continue from where you were; everything you started is intact. Never end your turn to wait.
