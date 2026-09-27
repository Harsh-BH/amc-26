# P57-iu-transfer — RESULT (final, 06:50)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Keep sub54's India/US. Do not revert to #34 or #8c, and there is no sub57.** On an honest frame (full density, every
  competitor p out of sample), sub54's recipe is **+0.000538 ± 0.000044 LB-eq over #34** (cross-fit; h0→h1 +0.000596 ± 0.000066,
  h1→h0 +0.000479 ± 0.000060). India +0.00067, US +0.00059, each ≥ 5 SE in both directions. Over #8c: +0.00073.
- In-sample competitor p is real but small. It inflates P44's holdout frame by 11% for sub54 (+0.000604 → +0.000538) and by 25%
  for the no-CE sub44. #34's own gain is untouched. An honest re-fit adds only +0.00002, which is below the 3 SE gate.
- The master's density / hn checks also clear sub54: test-shape re-weighting (+0.00067), P62's distractor clones (+0.00067), share
  features (tie), test adds' hn mix (expected precision 0.84 / 0.82). **The LB shortfall is not in India/US's holdout → test transfer.**
  The next suspect is sub54's France stack; the France-only diagnostic decides it.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Final India/US = sub54's** (P50 `fullzoo` boosters, `data/work/p50/prob_test_{c}_fullzoo.parquet`).
  - No change to the file or the code path.
  - The ranking on honest data is sub54 recipe (≈ honest refit) > sub44r > rall_only > sub44 > #34 > #8c.
- **Do not build sub57.** On the honest frame, the refit is +0.000559 vs the orig recipe's +0.000538 (+0.00002, ≪ 3 SE).
  It would be another full-holdout refit of the same recipe.
- **Do not switch to relative (share) features.** rel54 ties sub54 on every frame (sw +0.000530 vs +0.000538; test-weighted +0.00063
  vs +0.00067). Its test flip profile deviates from holdout exactly as sub54's does.
- Expected LB of sub54's India/US over gc70's (#34): **≈ +0.0004 to +0.0006**.
  - Scaling: honest +0.00054; test-shape weighted +0.00067.
  - Test flip rates are ×0.75 for adds and ×1.1–1.35 for drops. Drops stay profitable: 65–67% of them remove a false pair; break-even ≈ 33%.
  - So India/US should account for all of sub54's +0.00038 LB gain over gc70, or more. That puts France's net at ≈ −0.0002 to 0.
- **Suggested diagnostic upload:** sub54 with France lines from #37_gc70, India/US kept.
  - Its LB minus gc70's is India/US's LB contribution; sub54's LB minus its LB is France's.
  - If it scores ≥ sub54, sub54's France stack is net negative, and this file is the better final.

## Evidence         (table: ledger name | world | key metrics | verdict)
All deltas are per-S1 F0.5 vs #34 (P34's a10, out of fold, re-derived in each world), LB-eq = 0.468 India + 0.383 US.
- "orig" = the recipe fit on half h of the full world (P44's frame, what shipped), scored on half 1−h of the world.
- "refit" = fit on the world's own half h.
- x-fit = the mean of both directions (disjoint report halves).
- The decision is acc = am_world & p ≥ 0.71 everywhere.

**Worlds** (`p57.py`; each re-derives argmax, a10 columns, #34's p', CTX2 / CTX3 inside the world):
| world | construction | honest? | competitor rows / argmax pair India, US | heavy mass (≥ .5) India, US |
|---|---|---|---|---|
| full | P44's frame (0 parity mismatches) | folds 2–5 competitors in-sample for m2 | 0.088, 0.121 | 0.0051, 0.0086 |
| **sw** | full, with folds 2–5 competitor p swapped to P45's `fsub` (fit on folds 6–9): p_m2 := fsub, logit p += logit fsub − logit p_m2 | **yes, all competitor p out of sample** | 0.082, 0.111 | 0.0046, 0.0079 |
| swi | the mirror: folds 6–9 swapped to fsub (in-sample for them), so 2–9 are all in-sample | control | 0.080, 0.109 | 0.0041, 0.0075 |
| c6 | S1 of folds 2–5 dropped together with the SX they own (a smaller universe) | yes; US density ≈ test | 0.052, 0.064 | 0.0023, 0.0032 |
| c4i / c4o | clusters of folds {0,2–5} / {0,1,6–8}: equal density, in- vs out-of-sample | control | 0.039 / 0.043, 0.049 / 0.051 | 0.0017 / 0.0018, 0.0021 / 0.0022 |
| oos6 | option (a) literally: folds 2–5 S1 rows dropped, their SX kept | **biased** (see What failed) | 0.068, 0.090 | 0.0027, 0.0040 |
| **test** | – | – | 0.077, 0.055 | 0.0037, 0.0035 |

fsub is a clean stand-in for m2: its out-of-sample competitor log-loss on folds 2–5 is 0.111 India / 0.130 US. m2's
out-of-sample log-loss on folds 6–9 is 0.109 / 0.128. The two logits correlate at 0.992. The holdout world's competitor rows
carry **no CE score** (r is NaN on every non-holdout row).

**Main table: cross-fit LB-eq vs #34 (India x-fit / US x-fit), per world**
| arm | full | **sw (honest)** | swi (more in-sample) | c6 | c4i | c4o |
|---|---|---|---|---|---|---|
| #34 vs #8c | +0.000193 | **+0.000192** | +0.000187 | +0.000346 | +0.000448 | +0.000416 |
| rall_only (P45 r_all) | +0.000278 | +0.000276 ± 0.000032 | +0.000277 | +0.000282 | +0.000276 | +0.000296 |
| sub44 (P44) | +0.000258 | +0.000193 ± 0.000036 | +0.000302 | +0.000251 | +0.000339 | +0.000270 |
| sub44_psumoff | +0.000162 | +0.000121 ± 0.000032 | +0.000191 | +0.000163 | +0.000210 | +0.000192 |
| sub44r | +0.000580 | +0.000499 ± 0.000044 | +0.000606 | +0.000567 | +0.000631 | +0.000586 |
| sub44r_psumoff | +0.000487 | +0.000454 ± 0.000040 | +0.000510 | +0.000464 | +0.000519 | +0.000500 |
| **sub54 recipe (FINAL + r_all + zoo)** | +0.000604 | **+0.000538 ± 0.000044** (+0.000668 / +0.000587) | +0.000642 | +0.000590 | +0.000646 | +0.000602 |
| rel54 (shares, not counts) | +0.000585 | +0.000530 ± 0.000043 | +0.000649 | +0.000599 | +0.000674 | +0.000617 |
| refit sub54 on the world | – | +0.000559 ± 0.000043 | +0.000654 | +0.000682* | – | – |

\*c6's refit learns c6's shrunk candidate lists: it adds 2–3× more pairs per accepted pair than test does. Not transferable.

**sub54 recipe on sw, per country and direction** (vs #34; vs #8c in brackets):
- fit h0 → h1: India +0.000763 ± 0.000116 (+0.000930), US +0.000624 ± 0.000097 (+0.000962).
- fit h1 → h0: India +0.000573 ± 0.000106 (+0.000686), US +0.000551 ± 0.000086 (+0.000872).
- #34 vs #8c on sw: India +0.000113 ± 0.000086 / +0.000167 ± 0.000103, US +0.000322 ± 0.000076 / +0.000338 ± 0.000078.

**In-sample effect** (sub54 recipe, cross-fit):
- sw (0 in-sample competitor folds) +0.000538 → full (4) +0.000604 → swi (8) +0.000642.
- At equal density: c4i − c4o = +0.000044.
- So in-sample p adds ≈ +0.00005–0.00007, which is 10–15% of the gain. It is larger for the no-CE sub44 (25%: sw +0.000193 vs full +0.000258).

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| competitor-calib-by-fold | holdout, gt labels | #8c p log-loss on competitor rows: folds 2–5 0.072 / 0.093 vs folds 1, 6–9 0.089 / 0.110 (India / US); no CE r on any competitor row | keep (mechanism real) |
| competitor-density-vs-test | worlds + test | US holdout 2.2× test's competitor rows. Cause: the train US pool (1.32M S1) vs test (~658k), not in-sample p (c4i ≈ c4o mass) | drop (hypothesis) |
| world-full-repro | full | reproduces P44: sub44 +0.000306 ± 0.000048 / +0.000210; sub44r +0.000656 ± 0.000062 / +0.000505 | keep |
| world-sw-swap-honest | sw | sub54 +0.000538 ± 0.000044; #34 vs #8c +0.000192 | **keep (the answer)** |
| world-swi-swap-control / c4i / c4o | controls | in-sample inflation +0.00004–0.00007 on sub54 | keep |
| world-c6-cluster | c6 | sub54 +0.000590 ± 0.000044 | keep |
| honest-refit-sub54-on-sw | sw | refit +0.000559 vs orig +0.000538 | drop (below the 3 SE gate) |
| rel-arms-shares-not-counts | all worlds | rel54 − sub54: full −0.00002, sw −0.00001, c6 +0.00001, c4o +0.00002 | drop (tie) |
| density-weighted-holdout | full / sw re-weighted to test's (argmax, accepted) counts per S1 | sw: sub54 +0.000538 → **+0.000671 ± 0.000093**; #34 vs #8c +0.00019 → +0.00037; rel54 +0.00063 | drop (test shape helps, it doesn't hurt) |
| hn-mismatch-sub54-flips-test-vs-holdout | test + holdout | India adds mismatch 0.186 test vs 0.196 holdout. US 0.381 vs 0.30 (the extra is "other" 0.171 vs 0.106), but holdout mismatch adds are 0.93–0.95 precise. Expected precision at test's hn mix: adds 0.843 India / 0.821 US (holdout 0.843 / 0.805); drops keep 0.35 / 0.33 true (profitable) | drop (no failure mode) |
| test-flip-transfer-rel-vs-abs | test vs sw h1, same h0 boosters | test/holdout adds × drops: sub54 India 0.75 × 1.15, US 0.73 × 1.27; rel54 0.70 × 1.06, 0.66 × 1.29; sub44r 0.69 × 1.21, 0.71 × 1.23 | drop (rel no closer) |
| world-oos6-S1rows | oos6 | #34 vs #8c −0.00044; sub54 orig +0.000331; its refit +0.00104, learned from orphans | drop (artifact) |

External corroboration (P62 message, 06:40): my full frame padded with Poisson(0.9) clones of every free SX, to test's
distractor density. sub54 +0.000730 / +0.000604 (vs +0.000681 / +0.000528 unpadded); #34 vs #8c shrinks 35%.

## What failed      (approaches tried and dropped, with the number that killed them)
- **The brief's option (a) as written** (restrict competitors to folds 0 + 1, or 0/1/6–9) is biased.
  - The dropped S1's own true SX stay in the world with no owner, so they look free and a combiner adds them. Test never has these orphans.
  - On oos6, #34 falls to −0.00044 vs #8c, and a refit "wins" +0.001 by learning to drop the orphans.
  - The cluster worlds (drop the S1 and its SX together) and the swap worlds fix this.
- **The in-sample-competitor explanation of the LB gap.** It accounts for ≤ 15% of sub54's holdout gain (≈ −0.00007 LB-eq).
- **Density as the explanation.** Test-shape re-weighting and P62's clone padding both *raise* the r_all combiners' gain.
  The flip-profile deviation (fewer adds, more drops on test) is the same with share features, so absolute counts do not drive it.
  P45 traced the extra drops to CE r_all scores below logit −4 on test.
- **An honest refit (sub57).** +0.00002 over the shipped recipe on sw. Not built.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
Run from the repo root with `.venv/bin/python`. Work files go to `data/work/p57/`. Cap 1.5 GB: run one step at a time.
- `p57.py build <c> [rows|cluster|swap]` writes `A_{world}_{c}.parquet`.
  - `world(d, …, folds, cluster, swap)` is the one sub-world builder. The full world asserts parity with P44's A frame.
  - About 2–3 min and 1.2 GB per country.
- `ev57.py <world>`: every arm, orig and refit, both directions.
  - Run `full` first. It fits and caches the recipe boosters `b_<arm>_h<h>_s<i>.txt`.
  - `P57_ARMS` / `P57_REFIT` select arms. About 4 min and 1.3 GB per world.
  - `ARMS`, `relcols()` (the share features) and `score()` (= te44.combine_p) live here.
- `sum57.py [world …]`: the tables above, from `ev57_<world>.json`.
- `dw.py`: the test-shape re-weighted deltas → `dw.json`.
- `hnmix.py`: the hn class mix and precision of flips → `hnmix.json`.
- `tt.py`: the arms on test (chunked; r_all_test / zoo_test are positional with P50's T frames), test/holdout flip ratios → `tt.json`.
- `diag0/1/2.py`: fold composition, competitor calibration, fsub vs m2. `dens.py`: test density. `tprof.py`: staged subs' test flip rates.
- `logall.py`: the ledger rows.
- Disk: the world frames are about 130 MB each. c2, c5, c4i, c4o, oos* were deleted after scoring; their results are in the JSON.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / rules:** analysis only (LightGBM, pandas). Nothing new ships.
- **The swap world's fold 2–5 competitor p are an approximation.** p is shifted in logit by fsub − m2, so the US P17 stack and the
  e5 band blend on top of m2 are kept, not recomputed.
- **One residual holdout ≠ test difference no world removes:** on test, competitor p includes the bge CE blend; on the holdout,
  competitor band rows carry an e5 blend. If anything, test's competitor signal is *sharper*.
- **Holdout vs LB:** the LB is a public subset. A +0.0005 India/US effect should show at about ±0.00006 there, so the shortfall is
  not India/US noise unless test differs in a way no label-free check here sees.
- **Assumptions:**
  - The "sub54 recipe" is evaluated half-split. The shipped boosters are the full-holdout refit, which P50 measured at +0.00004 more.
  - LB weights are 0.468 / 0.383.
  - France is untouched by this probe.

## Open questions   (for the master; each with your best guess)
- **Where did sub54's missing ~+0.001 go?** Best guess: France. sub54's France stack (P43 recipe levers, P46, P48 veto, P54 rescue)
  is LOCO-estimated only, while India/US transfers at ≥ 75–90% here. Upload sub54 with #37_gc70's France lines to measure it.
- **Final India/US if the diagnostic shows India/US ≈ 0 on the LB?** Still sub54's. Every honest-frame, density and hn check puts it
  ahead of #34 and #8c, and #34 itself was ≈ flat on the LB (#34_veto +0.00005). A near-zero LB reading would point to a test shift
  that none of these frames can measure, and reverting would not recover anything.
