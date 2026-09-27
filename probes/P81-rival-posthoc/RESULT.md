# P81: the rank-above team's post-hoc rules on our sub84

The source is github.com/PundarikakshNTripathi/present-day-ber-amlc-2026 (MIT, public LB 0.99037). The gain in their last
LB step, 0.98850 → 0.99037, came from France post-hoc rule passes over a finished submission (`scripts/fr_posthoc{,2,3}.py`).
Their scripts are run as-is on our file, after their `ber.normalize` (v6) on test. None of their code is copied into src/.

## Holdout test (labelled, train fold 0, the p57_sub37 holdout frame, their rules gated to India or US)

| rule set | India F0.5 | US F0.5 | adds (precision) | removals that were true pairs |
|---|---|---|---|---|
| their full chain, pass 1 (S,V,W,a,b,b2,c,d1,d2g) | −0.000005 | +0.000046 | 278 (.76) / 353 (.93) | 43/44, 273/274 |
| their pass 2 (W,S2,F,E,D) | **−0.004688** | **−0.009712** | .85 / .85 | 5,123/5,128, 14,839/14,841 |
| their pass 3 (all F-rules) | −0.001084 | −0.000047 | — | 935/958, 88/91 |
| **variant A = their adds only, no c / F6, F7 guard** | **+0.000759** | **+0.001153** | 1,653 (.852) / 3,540 (.858) | 1/2, 1/1 |
| pass 2 adds only (F,E,D) | +0.000662 | +0.000951 | .848 / .849 | — |

- Their drop rules remove true pairs. Their sector swap W, run over **all** holdout candidates, not only predicted ones,
  flags 13,986 US / 4,698 India pairs, and 99.5–100% / 87–100% of them are true in every legal-form position.
  - The words are the generator's noise fillers: US center/service/partners.
  - On our France file, 90% of W's drops are the French analogues centre/service.
  - All drops (S, V, W, S2, F1, F3, F4*, F5R) are therefore left out.
- Rule c is out (India precision .37). F6 (legal-form change at site) is out because it contradicts P64's LOCO-validated veto.

## Test files (on `output/sub84x` = `final_patch.py` over sub76_brand; sub77's exact P31 frame is not on this laptop)

- `cs81A.parquet`: +24,029 / −1 (France +21,667, India +1,815, US +547).
  - France adds: ph1 a 1,205 · b 2,680 · b2 648 · d1 1,694 · d2g 667; ph2 D 795 · E 1,965 · F 1,804; ph3 F5A 4,954 · F2 5,334 · F8 13.
  - F7 removed 79 of their own adds.
  - India/US: pass 2 D/E/F adds only. Their CAP removals are not taken, because our lines list duplicate copies past 5/6 on purpose.
- `cs81A12.parquet`: cs81A without pass 3 (their LB-backed part: ph1 + ph2): +13,728 / −1.
- Both applied (`apply_cs.py apply`, one owner per SX, candidates extended): validator `--check-ids` PASS for
  `output/sub84x_81A` (md5 033b403b…) and `output/sub84x_81A12` (md5 2a9bd074…).
- On the lead's sub84: `python probes/P81-rival-posthoc/apply_cs.py apply output/sub84 probes/P81-rival-posthoc/cs81A.parquet output/sub84_81A`.

## Price (holdout per-add value: a true add gains ≈ .08–.11 per S1, a false add costs ≈ .2–.37; break-even precision ≈ .75)

| file | France precision .75 | .85 | .95 |
|---|---|---|---|
| 81A (+21,667 FR, +2,362 IU) | ≈ +0.00005 | **≈ +0.0005** | ≈ +0.0010 |
| 81A12 (+11,366 FR, +2,362 IU) | ≈ +0.00005 | ≈ +0.0003 | ≈ +0.0006 |

India/US contribute about +0.00005 of that. Our final India/US model already holds most E pairs: 663 E adds on test India,
against about 13k that holdout proportions predict.
