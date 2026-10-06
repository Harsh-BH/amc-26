# P6-france — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
Dense LOCO: a GBDT trained on one country loses 5–7 pts F0.5 on the other, and it is over-confident there. US→India scores
0.9054 vs 0.9792 in-country, India→US 0.9296 vs 0.9808, and the best t moves from 0.72 up to 0.87–0.98. Label-free FS-EM is worse (0.27–0.75).
**Predict France with the pooled US+India GBDT (all features) at t_France = 0.82 for P4's M2 (t* 0.71).** The threshold is
where France's predicted empty-S1 rate equals the US+India test rate at t*. In LOCO this rule recovers the oracle (0.9054 = oracle, 0.9293 vs 0.9296).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **Model:** the pooled US+India model with the full P4 feature set. Country stays out of the features (it already is).
   No feature group earned a France-specific drop. Dropping state, chain or city is neutral (±0.2 pt). Dropping context
   costs 0.7–0.8 pt cross-country.
   House-number features are the one country-specific group: they make the US model over-confident on hn-poor India,
   and dropping them gives +0.75 pt US→India. France carries house numbers like the US (91% of S2/S3, and they resolve
   64% of its chain ambiguity), so keep them.
   The transfer loss is a joint, country-level recalibration. The threshold fixes most of it; features don't.
2. **Threshold (the lever that matters):**
   - Global t* for US/India, as P4/P5 sweep it.
   - For France: `t_France = fr_diag.france_threshold(empty_curve_France, ref_empty, t_star)`. Here ref_empty is the model's
     S1-weighted US+India **test** empty-S1 rate at t*, and the result is clipped to [t*, t*+0.2].
   - One command on the final model's test predictions (s1, sx, p per country):
     `.venv/bin/python probes/P6-france/fr_threshold.py 'data/work/p4/prob_test_{c}.parquet' <t*>`
   - **P4's final M2 (t* 0.71): t_France = 0.82** (assigned-share cross-check 0.84). France at t* is over-matched:
     5.23% empty vs the 5.69% reference, 60.3% of S2/S3 assigned vs 58.1%. The LOCO pooled model agrees
     (t* 0.73 → 0.83 / 0.84).
   - If the procedure cannot run, use **t* + 0.10**. That costs ≤ 0.07 pt in-country (pooled: US 0.9807→0.9800,
     India 0.9782→0.9776) and gains +0.9 pt (US→India 0.8636→0.8723) and +0.3 pt (India→US 0.9260→0.9287) on an
     unseen country.
3. **Expected France output at t_France** (sanity check on the submission; P4 M2, test France at t 0.82):
   - empty S1 5.68% (US/India test at t* 5.73% / 5.66%);
   - 3.23 matches per S1 (US/India 3.36 / 3.37);
   - 58.4% of S2/S3 assigned (US/India 58.4% / 57.9%);
   - list sizes 1 / 2 / 3 / 4 / 5+ = 8.5 / 19.3 / 24.4 / 20.5 / 21.6% (US 6.5 / 18.0 / 24.4 / 21.3 / 24.1%).

   Red flags: France empty rate < 4.5% or > 7.5%, or matches per S1 < 2.9. Either means the France stage broke
   (normalisation, candidates or threshold), not France being different. US/India test reproduce their holdout
   profile to ±0.1 pt (pooled LOCO model: 5.74% vs 5.73% US, 5.65% vs 5.65% India).
4. **Normalisation:** fold `france.py` (filler=False) into `normalize.py`. It adds three France-only rules:
   - parenthesised house numbers `(520) RUE X`, generic but only seen in France (1.6% of France S2/S3);
   - French street/building segments (`R. VICTOR HUGO`, `PLACE …`) are not places/city;
   - St/Ste → Saint/Sainte in France names.

   US/India output is unchanged (asserted). Label-free rule-stage effect on France test: +7.6k accepted pairs,
   −0.6k lost, S1 coverage 0.8642 → 0.8664; S2/S3 house-number parse 91.5 → 93.1%, city in the 15 real cities 94.6 → 96.6%.
5. **Fallback:** if France's profile trips a red flag, fall back to France = rule-accepted pairs ∩ p ≥ t*. Never predict
   France empty: the all-empty baseline is 0.056.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| loco-dense-base | holdout-dense-full | in-country US→US 0.9808 (t .72), India→India 0.9792 (t .72); pooled US 0.9807 / India 0.9782 (t .73/.74). Cross US→India 0.9054 at oracle t .98, 0.8636 at source t, 0.8507 at t .5; India→US 0.9296 (t .87), 0.9260 at source t | keep |
| loco-dense-threshold-policies | holdout-dense-full | unseen target: empty-rate match US→India 0.9054 (= oracle), India→US 0.9293 (oracle 0.9296); matches/S1 match 0.9021 / 0.9285; t .85: 0.8761 / 0.9294; t .90: 0.8843 / 0.9294. In-country pooled t .85: −0.11 / −0.09 pt, t .90: −0.28 / −0.30 pt | keep |
| loco-dense-decompose | holdout-dense-full | loss spread evenly: US→India Latin S1 0.9014 vs cross-script 0.9144 (in-country 0.9763 / 0.9857); India→US chain 0.9222, no-chain 0.9366 (in-country 0.9749 / 0.9864) | keep |
| loco-dense-drop-{state,chain,city} | holdout-dense-full | every change is within ±0.2 pt of base. state: 0.9808 / 0.9789 in-country, 0.9075 / 0.9293 cross; chain: 0.9807 / 0.9787, 0.9077 / 0.9287; city: 0.9809 / 0.9789, 0.9039 / 0.9299 | neutral, keep all |
| loco-dense-drop-hn | holdout-dense-full | in-country US 0.9790 (−0.18), India 0.9789; cross US→India 0.9129 (+0.75; at source t 0.9051 vs 0.8636), India→US 0.9267 (−0.3) | hn drives US-model over-confidence on hn-poor India; France is hn-rich (91%), keep |
| loco-dense-drop-context | holdout-dense-full | in-country −0.3 / −0.4 pt, cross −0.7 / −0.8 pt (US→India 0.8981, India→US 0.9219) | context transfers |
| loco-dense-selftrain | holdout-dense-full | source labels + source-model pseudo-labels on target fold 2 (p ≥ .98 argmax / ≤ .02; 98.4% / 99.3% label accuracy): US+psIndia→India 0.8701 (plain 0.9054, pp 0.888), India+psUS→US 0.9311 (plain 0.9296) | drop |
| loco-dense-frshape-plain | holdout-dense-full | France-shaped scoring (35% of SX lose their state): US→US 0.9808, pooled US 0.9807 / India 0.9782, identical to base; US→India 0.9058 (base 0.9054) | no risk, no augmentation needed |
| fs-em-vs-gbdt-dense-US>India / India>US | holdout-dense-full | FS-EM fitted on the target unlabelled: 0.2699 / 0.2723 (EM collapses to λ 0.83 / 0.80); GBDT cross 0.9054 / 0.9296 | GBDT |
| fs-em-fixed-lambda-dense | holdout-dense-full | λ fixed 3.4/14.1: India 0.7496, US 0.6811 | drop |
| fs-em-vs-gbdt-France-labelfree | test-France | FS-EM converges on France (λ 0.30); vs pooled GBDT: 672k SX decisions agree, 201k FS-only, 192k GBDT-only | FS unsafe |
| france-uncertain-band-pooled | test | share of SX whose argmax p ∈ [0.3, 0.9): France 9.9% vs US 3.1% / India 4.3%; p ≥ 0.9: 56.6% vs 57.3% / 56.1% | France has 2–3× the uncertain mass |
| france-threshold-prob_test_{c} | test | P4 M2: ref empty 5.69% at t* .71; France 5.23% at t* → empty match t .82, assigned match t .84 | keep |
| france-threshold-pooled | test | ref empty 5.69% at t* .73; France 5.27% at t* → empty match t .83, assigned-share match t .84 | keep |
| loco-dev-drop:none | dev | US→US 0.9930, India→US 0.9912; India→India 0.9922, US→India 0.9899 | dev world too easy: it hides the 5–7 pt loss |
| fs-em-vs-gbdt-US>India / India>US | dev | FS-EM 0.9588 / 0.9701, M0 0.9653 / 0.9757, GBDT cross 0.9899 / 0.9912 | GBDT |
| fr-layer-nofiller-rules-labelfree | test-France | +7.6k / −0.6k rule pairs, S1 coverage 0.8642→0.8664 | keep |
| fr-layer-rules-labelfree | test-France | with FR_FILLER: +31.6k / −18.4k, S1 coverage 0.8642→0.8611 | FR_FILLER off |

France structure (label-free, test):
- **Names:** France is India-like. 30% of S1 share core+city (India 22%, US 1.2%). Names are short (2.05 core tokens).
  17% of S2/S3 have ≥ 2 exact-core S1 candidates (US 11.5%, India 22.7%).
- **Addresses:** France is US-like. S2/S3 carry a house number on 91% of rows (India 58%), and a unique house number
  resolves 64% of the chain ambiguity (US 60%, India 43%).
- **France-only quirks:** S2/S3 carry a state on only 65% of rows (US/India 97%), and there are only 15 cities (99% of S1).

## What failed      (approaches tried and dropped, with the number that killed them)
- **Source-country threshold on an unseen country:**
  - US→India 0.8636 vs its oracle 0.9054 (−4.2 pt); India→US −0.4 pt.
  - Unseen countries get over-matched at low t: at t .3 the empty-S1 rate is 2.2% vs a true 5.5%.
  - France shows a milder version: 5.27% empty at t* vs the 5.69% reference.
- **Label-free Fellegi-Sunter (Splink's model) as the France scorer:**
  - Free EM collapses on dense candidates (λ → 0.8, F0.5 0.27).
  - With λ fixed it reaches 0.68–0.75, below the cross-country GBDT (0.905/0.930) and the rules.
  - It disagrees with the GBDT on ~200k France S2/S3 each way.
  - Not run through the splink package: P4's fs_fit is the same m/u EM model. Splink's random-pair u estimation might
    fix the collapse, but not a 15-pt gap.
- **FR_FILLER** (dropping fils/frères/cie/compagnie/associés from France cores):
  - +31.6k rule pairs but −18.4k lost, and S1 coverage falls.
  - France S1 names really contain "& Fils", so dropping the suffix merges distinct S1 into ties. Opt-in flag, off.
- **Self-training on the unseen country's confident pseudo-labels (LOCO):**
  - US+pseudo-India → India 0.8701, vs 0.9054 without; precision falls to 0.888. India+pseudo-US → US: +0.15 pt.
  - The 1.6% wrong-but-confident pseudo-positives teach the model to trust exactly its mistakes. Don't self-train on France.
- **Dev-world LOCO:** it hides the whole effect (≤ 0.23 pt loss on dev vs 5–7 pt dense). Never judge transfer on dev.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `france.py`:
  - `name_core(s, country, filler=False)`, `address_fields(addr, country, freq=None)` (normalize.address_fields
    + France fixes), `feats(df)` (= rules._feats columns);
  - self-check `python france.py`. France test (1.7M rows): 2.5 min single process, 1.5 GB.
- `fr_diag.py`:
  - `profile(q1, qx, p, n_s1, n_sx, t)`, `anchor(p, rule_acc, t)`, `empty_curve(q1, qx, p, n_s1)`,
    `france_threshold(fr_empty, ref_empty, t_star, cap=0.2)`;
  - self-check included.
- `fr_threshold.py <model|'path_{c}.parquet'> <t*>`: prints the empty / assigned curves per country, logs t_France. It reads
  (q1, qx, p) from `fr_score.py` or any (s1, sx, p) test file. It runs in 2 min and 1 GB.
- `fr_score.py <booster.txt> <country…>`: scores P4-format test X → `data/work/p6/prob_test_<c>_<model>.parquet`.
  France 95 s, US 8.5 min (10.4M pairs), 1.2 GB.
- `loco_dense.py base|drop <groups>|frshape|selftrain`: dense LOCO on P4's train X, built on P4's `train.load/score_folds/truth_all`.
  Models are saved to `data/work/p6/models/{US,India,pooled}.txt` (fold 2 ×0.5, early-stopped on fold 1 of the source).
  Takes ~4 min per model scored, 1.8 GB.
- `ho_profile.py`, `decompose.py`, `policy.py`, `fs_dense.py`, `fs.py`, `loco.py` (dev): the analyses above.
- `p4x.py`, `fr_x.py`: P4's featrun stages redirected to `data/work/p6/p4out/X`. They hold test X for France / US / India in
  P4's exact format (1.0 GB), plus train US / 2 partial India files (0.8 GB, deletable: P4's own train X is complete).
  Reusable by P4/P5 if their feature code is unchanged.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Assumption behind t_France:** France's true empty-S1 rate equals US/India's. The same generator makes this plausible:
  singleton share is 5.58 / 5.59% in train, and test US/India reproduce the holdout profile. If France truly had more
  singletons, the matched t would be too high. The [t*, t*+0.2] clip bounds the damage to the in-country slope:
  −0.1 pt at t*+0.1 and −0.5 / −0.6 pt at the cap (pooled US / India).
- **The two LOCO directions are the only unseen-country evidence.** France with a pooled model is "less unseen" than
  either direction: its empty-rate gap at t* is 0.42 pt vs 1.0 (India→US) and 2.7 (US→India). Expect a smaller loss
  than LOCO and a smaller threshold gain.
- **The LOCO models are mine** (fold 2 ×0.5, lr 0.1). Rerun fr_threshold.py on P4's final predictions: t* and t_France move with calibration.
- **Licences:** no new dependency. france.py uses only normalize.py (anyascii, ISC) and pandas; the models are
  LightGBM (MIT). No external data.
- **No test-row fitting.** France rules come from domain knowledge. Test rows were only counted (coverage, token asymmetry) to check them.
- **The master re-based the branch mid-run.** src/ gained run.py/model.py, which shadow P4 modules; P4 renamed run.py→featrun.py.
  My scripts import P4's modules from `../p4-match` by path, so they break if that worktree moves.
- **Chain counts scale with pool size:** France S1 pool 259k vs India train 883k. Dropping chain_1/chain_x is neutral
  in LOCO (±0.1 pt), so the scale mismatch cannot hurt much.

## Open questions   (for the master; each with your best guess)
- **Can place_freq be fitted on test S1 (open question 4)?** Best guess: unnecessary for France now. The street-segment
  fix puts 96.6% of France S2/S3 cities in the 15 real cities.
- **Should France get its own model stage (e.g. P5's decision layer fitted on US+India)?** Best guess: no. Use the same
  layer and change only the threshold per country, via fr_threshold.py.
- **Where should P4b/P5 spend effort?** France has 2–3× the uncertain band of US/India (9.9% of its S2/S3 at p 0.3–0.9).
  Best guess: a re-ranker on the uncertain band pays most on France. It must be trained on US+India only and validated
  by LOCO: self-training on France pseudo-labels was tested by LOCO and loses 3.5 pt.
