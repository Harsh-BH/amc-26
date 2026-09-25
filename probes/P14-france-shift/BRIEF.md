# P14-france-shift: close France's gap by fixing what makes France look different from training
Read first: docs/PROBLEM.md, probes/P6-france/RESULT.md + france.py (France rules: street types, St/Ste, "(520) RUE",
régions/départements), probes/P12-synth/RESULT.md (synthetic data failed: noise RATES must match the target),
SUBMISSIONS.md (D1/D2/D4 rows), code/business_entity_resolution/README.md. Own worktree; code under
probes/P14-france-shift/ (src/ is owned by P9 — propose src changes in RESULT.md, do not edit src/).

**Facts (master, 25 Sep 23:10, from the public leaderboard):**
- #5 LB 0.980017. France-emptied 0.849163 ⇒ France F0.5 ≈ 0.931, India+US ≈ 0.989 (= our holdout).
- France threshold: t 0.78 → 0.980017, t 0.88 → 0.981024, t 0.95 → 0.981142 (flattening; best so far).
- Top of the LB is 0.988419. If their India+US ≈ ours, their France is ≈ 0.98+. **The whole gap is France.**
- **Distribution shift found:** among France S2/S3 records, **35% have an empty parsed `state`** (US 2.9%, India 2.4%).
  Those addresses end in a city only ("36 RUE DÉSIRÉ BONDUES, LILLE"). France S1 has only 3 régions
  (hauts de france, nouvelle aquitaine, pays de la loire) and ~15 real cities. The matcher and blocking were trained
  where a missing state is rare and meaningful — on France it is routine. Also: France city agreement on confident
  matches is 0.95, house number 0.89 (US 0.82 / 0.81), so France addresses are actually MORE informative.

**Goal:** a France prediction that is measurably better on the leaderboard, with a label-free argument for why.
Candidate levers, measure each:
1. **Fill missing state from city**, country-agnostic: the majority state of that city among the same split's
   records that have both (label-free, computed from the dataset itself; note in RESULT.md that it uses unlabelled
   test text — open question 4 — and offer a train-only variant if one exists, e.g. a mapping built from S1 only).
   Check US/India holdout is unchanged or better with the same rule, then rebuild France's blocking/features/scores.
2. **Missingness-matched training (train-only, allowed):** blank `state` on ~35% of train S2/S3 (the France rate)
   and retrain / or evaluate m2_big on a holdout with state blanked at 35%: how much does m2_big lose when state is
   missing that often? If a lot, a model trained with matched missingness should transfer better. This is the
   "match the target's rates" lesson from P12, applied to the one rate we can measure on France without labels.
3. P6's france.py rules (street types, St/Ste, parenthesised house numbers) — label-free coverage deltas.
4. Anything else you find by comparing France's feature distributions with US/India's (feature by feature, on the
   pairs the model accepts and on all pairs) — rank shifts by size.
Validate every lever on the US/India holdout (must not lose > 0.0003) and with a France proxy (e.g. holdout with
France-like missingness). Final France threshold: re-run P6's rule and report it next to 0.95.
Deliverables: RESULT.md with a ranked list; for the best 1–2 variants, France probability files
data/work/p14/prob_test_France_<variant>.parquet (s1, sx, p) that the master can combine with #6c's India/US for a
leaderboard test tomorrow (5 slots on 26 Sep). Memory cap 4 GB incl. CLI, swap allowed; disk ~8 GB — delete scratch.
Never end your turn to wait; detach long jobs and poll with bounded waits. Ledger every run.
