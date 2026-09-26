# P52-fr-calib: treat France's under-confidence as calibration — prior-shift EM, importance weights, adversarial feature ranks
Why: every France rule today recovered pairs at p 0.3–0.9 that are ≥ 95% true on train analogues; France runs on thresholds
transferred via LOCO (P42 joint tune: t 0.99 / rescue 0.90 / gc 0.70 / hn extras 0.80) and density matching. A literature
review ranks label-shift / covariate-shift correction as the most promising untried lever (Saerens 2002 EM prior
re-estimation; Alexandari et al. BCTS calibration; Park et al. AISTATS 2020 importance weighting; adversarial validation /
DADER-style feature alignment). Current best = output/sub48 (France = sub46_on43 France − P48's 1,138 low-CE vetoes;
India/US = P44 combiner + r_all); sub52_max (P50) is being built on it. Read probes/P42-fr-joint-tune (solution_p42.py
LOCO machinery), P37, P39, P48 RESULT.md, P47 RESULT.md (shape matching failed: pool top only 67–75% true). Own worktree.
Arms (LOCO both directions, vs the current recipe analogue, France-F0.5; ≤ 45 min each; report duds too):
1. EM prior re-estimation (Saerens) on BCTS-calibrated pair probabilities: estimate π_target from the target's unlabelled
   score distribution; shift logits by log-odds(π̂_t) − log-odds(π_s); re-derive thresholds. Check π̂ vs true π in LOCO.
2. Importance-weighted threshold choice: a domain classifier (source vs target pair features) → weights p(t|x)/p(s|x)
   on source holdout pairs → pick t (and rescue/gc bars) maximising weighted F0.5.
3. Adversarial validation: train source-vs-France classifier on the matcher/stacker features; AUC + top features; replace
   the top 5 shifted raw features with per-country quantile ranks (or drop), refit the France-chain scorer, LOCO.
4. Quantile-map France top-1 CE/p scores onto the source top-1 distribution, then apply source thresholds.
Ship rule: both LOCO directions ≥ 0 and India→US ≥ +0.0003 France-F0.5 (the LB has tracked India→US). Build output/sub52cal
on the current best staged file (sub52_max if it exists, else sub48; France changed only), validator --check-ids,
SUBMISSIONS.md row with expected LB. Memory 2 GB local; SageMaker CPU allowed ($10). DISK ~2 GB. Never end your turn to
wait. Deadline: arm table by 07:30, file by 09:00.
