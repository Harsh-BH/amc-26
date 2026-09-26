# P26-france-da — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The domain adaptation that works is house-number features inside the main matcher.** Importance reweighting and pruned-list context
  fail. LOCO at P6's label-free threshold: **+1.83 ± 0.04 pt US→India, +1.90 ± 0.03 pt India→US**.
  With the bge CE blended on top and everything out-of-domain (France's situation): **+2.01 ± 0.06 / +1.76 ± 0.04 pt**.
- In-country the matcher alone gains +0.0015 (India) / +0.0032 (US), but only +0.00009 ± 0.00006 once #8c's bge CE + US stacker are on top. India/US stay #8c.
- **Upload `data/work/p26/sub26_hn/`:** #8c India/US (byte-identical) + France from the hn matcher + bge CE at t 0.986 (D4's empty rate). Validator PASS. Spare: `sub26_hn_t975`.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **LB slot:** `data/work/p26/sub26_hn/{matching_results.tsv, candidate_pairs.tsv}` (add code.zip). Validator PASS.
   - India/US rows are byte-identical to `output/sub8c`. 25,578 S1 differ from #9ab, all of them France.
     `candidate_pairs.tsv` is byte-identical to #9ab's (P14's IDF list, #6 pruner, 5.03 per S1).
   - France file: `data/work/p26/prob_test_France_full_hn_bge.parquet` (s1, sx, p, p_gbdt).
     - p_gbdt = the hn matcher `c_hn`, scored on P14's IDF-fixed France X (`data/work/p14/w_idfx`) over the #6-pruned keys.
     - p = P13's blend at P13's out-of-domain France weights [0.9245, 0.2595, 0.0515], band 0.02–0.98.
     - All 331,663 band pairs have a bge r: 315,638 from P13/P19, 34,339 from P26's top-up job (control pairs: median |Δlogit| 0).
   - t 0.986 = D4's empty-S1 rate: 6.84% empty, 3.01 matches per S1 (#9ab at its 0.985: 6.71% / 3.02). P6's rule would give 0.715.
   - Rebuild:
     `.venv/bin/python probes/P14-france-shift/combine.py output/sub8c data/work/p26/prob_test_France_full_hn_bge.parquet 0.986 <out>`
   - How to read the LB: above 0.984072 (#9ab) ⇒ keep. France is ~15% of the LB, so +1 pt of France F0.5 ≈ +0.0015 LB.
2. **Spare slot (threshold step, only if sub26_hn beats #9ab):** `data/work/p26/sub26_hn_t975/` (t 0.975: 6.49% empty, 3.10 per S1).
   Validator PASS; 25,263 S1 differ from #9ab, all France.
   - The high France t mostly removed generated siblings (P15). The hn matcher accepts **3× fewer sibling-offset pairs than #9ab at every t**
     (0.55 vs 1.52 per 1k at 0.975).
   - P20's biggest cell (one core word swapped, hn equal; US: 89% true) is only ~30% accepted at the D4 threshold in every file.
3. **India/US: keep #8c.** On P22/P25's #8c yardstick, replacing m2_big with the hn matcher gives:
   - +0.000085 ± 0.00006 with the CE blend and US stacker refit;
   - −0.00008 ± 0.00007 with the shipped stacker;
   - ±0.00001 when averaged with m2_big.

   None reaches the +0.0003 at 3 SE bar. Without the CE it is +0.0023 ± 0.00014: in-domain, the bge CE and P17's stacker already
   carry the hn signal.
4. **src promotion (P9), if the upload wins:**
   - Function: `solution_p26.hn_feats(q1, qx, name_cos, M) -> float32[n, 5]`.
     - M = {h1, l1: hn value and digit count per S1 row; hx, lx: the same per SX row; −1 = none}.
     - The hn is rules' `rf_*` column `hn`, parsed to int (1–9 digits; see `prep.py`).
   - Compute it on each country's **full** candidate list (before the pruner), right after features.py assembles X.
     It appends `hn_cls, hn_d, hn_nsame, hn_nssame, hn_ns1hn` after the 65 features.
   - Matcher `c_hn.txt`: m2_big's recipe (US+India folds 2–5, lr 0.05, 127 leaves, early stop on fold 1 ×0.25) on the 70 features.
     3,579 trees, val logloss 0.0189 (m2_big 0.0229).
     Booster: `s3://sagemaker-114872709824-ap-south-1/amc26-p26/results/p26-full-hn-0926-1533/c_hn.txt`.
   - Retrain (15 min, 20 GB, ≈ $0.30):
     `probes/P26-france-da/launch.sh full_hn ml.m6i.4xlarge '{"mode":"full","lr":0.05,"rounds":8000,"es":100,"folds":[2,3,4,5],"arms":[{"name":"c_hn","hn":1}]}'`
   - Integration, two options:
     - **One matcher everywhere:** c_hn for all countries, with the India/US CE weights and US stacker refit (neutral: +0.00009 ± 0.00006).
     - **Minimal:** c_hn for France only, which reproduces this upload exactly.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| phaseA-adversarial-France-vs-train | test-France + train sample (116,952 pruned pairs per domain) | domain AUC France vs train **0.998**; France vs US 0.998, France vs India 0.999, **US vs India 0.998**. Same AUC on true-pair-like rows (m2 p ≥ 0.5: 0.998) and look-alikes (p < 0.5: 0.996). Dropping the top-10 shifted features still leaves 0.988. Band share (0.02 < p < 0.98): France 29%, train 16% | shift is everywhere; plain IW unusable |
| phaseA2-reweighting-feature-sets | same | weights p/(1−p), clip [0.1, 10]: all 65 feats AUC 0.998, ESS 0.064 (91% of rows at the floor); 29 pair-similarity feats ("sim") AUC 0.956, ESS 0.16; names-only (14) AUC 0.82, ESS 0.36; addresses-only (8) 0.919 / 0.19. India-vs-US on the same sets: 0.998 / 0.974 / 0.822 / 0.951 | LOCO tests "sim" (full, √-tempered) and "all" |
| loco-US>India-a0+a1+a1h+a1all+a3 / loco-India>US-(same) | holdout full-corpus fold 0, #6-pruned list | target F0.5 at P6's rule, Δ vs a0 (paired SE): **weights on 29 sim feats (a1) −0.0054 ± 0.0004 / −0.0101 ± 0.0003**; √-tempered (a1h, ESS 0.34–0.44) −0.0048 / −0.0071; all-65 weights (a1all, ESS 0.05–0.10) −0.0026 / −0.0063; weights + hn (a3) +0.0158 / +0.0152 (below hn alone). Source in-country: a1 −0.0016 / −0.0041 | **drop weighting** |
| loco-US>India-a0+a2+ak+actx+actx2 / loco-India>US-(same) | same | **hn in the main model (a2): 0.9213 → 0.9396, +0.0183 ± 0.0004 US→India; 0.9348 → 0.9539, +0.0190 ± 0.0003 India→US**; oracle t: +0.0183 / +0.0204; source in-country +0.0032 ± 0.0002 (US) / +0.0009 ± 0.0002 (India). Kept-list training (ak) −0.0047 / −0.0036; pruned-list context (actx, kept-list training) −0.0021 / +0.0008; actx2 (ctx + hn, kept) +0.0143 / +0.0237 | **keep a2** |
| loco-US>India-a2+a2s+a2sup+a2cf+a0cf / loco-India>US-(same) | same | a2 reproduces bit-identically (0.9396 / 0.9539). hn class + delta only (a2s) 0.9265 / 0.9546; support only (a2sup) 0.9340 / 0.9408, so both halves are needed. Pruned context with full-list training (non-kept rows NaN): alone (a0cf) 0.9182 / 0.9350 (≈ a0: −0.003 / +0.000); with hn (a2cf) 0.9379 / 0.9586 (vs a2: −0.0017 / +0.0047) | ctx: fails the gate alone; mixed with hn |
| (run-to-run noise) | same | the same a0 config on two instances: 0.9219 vs 0.9213 (US→India), 0.9347 vs 0.9348 (India→US); same instance type + seed = bit-identical | noise ≈ 0.0006 ≪ +0.018 |
| ceproxy-loco-gbdt+loco-bge-WOOD | holdout fold 0 half 1, pruned | France procedure with GBDT **and** CE trained on the source only: source blended in-domain, target at P13's WOOD, P6 rule clipped [t*, t*+0.2]. **a2+CE vs a0+CE: +0.0201 ± 0.0006 (US→India 0.9212 → 0.9413), +0.0176 ± 0.0004 (India→US 0.9405 → 0.9580)**. CE on a0 +0.0023 / +0.0051, on a2 +0.0102 / +0.0042. Unclipped rule: +0.0200 / +0.0173. LOCO r cover only ~25% of the LOCO target band (P13 scored m2_big's holdout band) | **hn gain survives the CE** |
| ceproxy2 (OOD blend weights for the hn base) | same | blend fit on the other direction's target (xfit, P13's `xw`) or the mean of both, vs WOOD, on a2: US→India −0.0035 ± 0.0002 / −0.0024 ± 0.0002, India→US +0.0032 ± 0.0002 / +0.0040 ± 0.0003 (a0 likewise: −0.0029 / +0.0055). a2 − a0 under xfit / mean: +0.0195 / +0.0191 and +0.0153 / +0.0128 | keep WOOD (refit is direction-dependent); hn robust to the weights |
| phaseC-full_hn (c_hn) | holdout fold 0 half 1, #6-pruned list, matcher only (no CE / stacker), t tuned on half 0 | vs m2_big on the same rows: **India 0.98274 → 0.98423, +0.00149 ± 0.00021; US 0.98337 → 0.98652, +0.00315 ± 0.00020** (at t 0.71: +0.00120 / +0.00308). 3,579 trees, val logloss 0.0189 (m2_big 0.0229), hn gain share 0.15. France checks: m2_big reproduces P14's France p exactly (max abs 0), and the pruner reproduces the France keys exactly | keep |
| eval8c-c_hn-India-US | holdout-dense-half1 (#8c yardstick, P22 ev.py) | c_hn alone: no CE +0.0023 ± 0.00014 vs the control; with CE + stacker refit **+0.000085 ± 0.00006 vs #8c** (India +0.00003, US +0.00012); shipped stacker −0.00008 ± 0.00007; avg with m2_big ±0.00001. Pair logloss India 0.0576 → 0.0518, US 0.0630 → 0.0456. The control reproduces #8c (0.98990 / h1 0.98956) | India/US: no change |
| phaseC-full_ctx2 / phaseC-full_cf | same as phaseC-full_hn | ctx + hn, kept-list training: India +0.00077 ± 0.00025, US +0.00310 ± 0.00021; ctx + hn, full-list training: India +0.00089 ± 0.00026, US +0.00330 ± 0.00021 (c_hn: +0.00149 / +0.00315) | below c_hn on India |
| bge-topup-France | test-France | 34,339 band pairs of the c_hn / c_cf files had no bge r; scored by P13's `bge_ckpt` (P19's recipe) on ml.g5.xlarge, 259 s (**$0.12**). 2,000 control pairs vs P13: corr 0.99996, median \|Δlogit\| 0, p99 0.125 | keep |
| profile-full_hn (France, label-free) | test-France | all 331,663 band pairs blended. **t_eq_d4 0.986: 6.84% empty, 3.01 per S1.** Accepted sibling-offset pairs per 1k (hx − h1 ∈ {3,…,21}) at t 0.90 / 0.95 / 0.975 / 0.985: hn 0.95 / 0.70 / 0.55 / 0.43 vs #9ab 2.91 / 2.02 / 1.52 / 1.06, at the same matches per S1 (±0.01). Accepted hn-equal share 0.899 (#9ab 0.901). P6 rule t 0.715 | → sub26_hn |
| taxonomy cells (P20's pairs_France) | test-France, argmax pairs | share accepted at each file's D4 t, #9ab → c_hn → c_cf: sub1 × hn-eq (86k) 0.300 → 0.295 → 0.295 (mean p 0.579 → 0.595 → 0.626); disjoint × hn-eq 0.480 → 0.458 → 0.555; core-eq × sib3 0.008 → 0.003 → 0.004; core-eq × other-hn 0.238 → 0.273 → 0.289; core-eq × hn-eq 0.966 → 0.962 → 0.962 | P20's cell is threshold-bound |
| profile-full_cf / profile-full_ctx2 (France) | test-France | c_cf: t_eq_d4 0.989, sib-offset 0.54 per 1k, 37,547 S1 differ from #9ab; ctx2: t 0.99, 0.44 per 1k, 39,279 differ | spares, not staged |

Phase A top shifted features (gain share in the France-vs-train classifier; univariate AUC; SMD France − train):
- place_n_1 0.19 (0.28, −1.04)
- sx_margin 0.17 (0.30, −0.68)
- place_inter 0.07
- core_ntok_1 0.06
- block_score 0.05
- name_len_1 0.05 (0.30, −0.78)
- state_both 0.05 (0.36, −0.71)

Record-shape features (place counts, name lengths, state presence) identify the country. So does sx_margin: P20 found France's
value is set by a competitor the pruner later removes.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Adversarial importance reweighting (the brief's arm 1): negative in every variant, in both directions.**
  - Why it can't work here: the 65 pair features separate every pair of countries at AUC 0.998, US vs India included.
  - All 65 features: ESS 5–10%, 91% of rows at the clip floor. LOCO −0.26 / −0.63 pt.
  - The 29 pair-similarity features (AUC 0.96–0.97, ESS 13–16%): −0.54 / −1.01 pt. √-tempered: −0.48 / −0.71 pt.
  - With hn features (arm 3): +1.58 / +1.52 pt, below hn alone (+1.83 / +1.90).
  - It also costs in-country: source −0.0016 / −0.0041.
- **Context features over the pruned list (P20's suggestion):**
  - Kept-list training costs −0.47 / −0.36 pt on its own.
  - Recomputing the context wins back +0.26 / +0.44, which is still −0.21 / +0.08 vs a0.
  - With full-list training: −0.3 / +0.0 pt alone, and −0.17 / +0.47 on top of hn.
  - On France it lifts the mean p of P20's cell (0.58 → 0.63) but not its acceptance at the D4 threshold.
  - Caveat: LOCO cannot reproduce France's IDF-driven competitor, so this is a weak "no".
- **hn features for India/US in the shipped pipeline:** +0.00009 ± 0.00006 on the #8c yardstick.
- **Refitting the out-of-domain blend weights for the hn base:** the sign depends on the direction (−0.0035 / +0.0032). WOOD stays.
- **Partial hn sets:**
  - class + delta only: −1.3 pt US→India vs full hn;
  - support only: −0.6 to −1.3 pt.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`solution_p26.py`**: the SageMaker CPU job, shipped as solution.py next to sm_train.py. Self-check: `python solution_p26.py`.
  - Core functions: `hn_feats`, `pruned_ctx`, `Country`, `Eval`.
  - `loco`: arms trained on the source's fold 2 → LOCO F0.5 at the P6 rule. 5 arms in 5–7 min, 13.6 GB.
  - `full`: m2_big's recipe + one arm → holdout paired vs m2_big, plus France p. 15–17 min, 20 GB.
  - `score`: a booster → P22-format holdout preds. 5 min.
  - `ceproxy` / `ceproxy2`: the France procedure with LOCO GBDT + LOCO bge. 4 min.
- **Launch and monitoring (bash + aws + jq):**
  - `launch.sh <tag> <instance> '<hp json>'`;
  - `ev_launch.sh`: P25's #8c eval on our score jobs;
  - `status.sh`, `logs.sh <tag>`.
- **Laptop side:**
  - `prep.py`: per-record house numbers + fold-0 truth → `data/work/p26/sm/meta/*.npz`. 30 s, < 1 GB.
  - `prep_r.py`: LOCO bge r → q1/qx.
  - `phase_a.py`, `phase_a2.py`: Phase A on samples. 2 min, 1.5 GB.
  - `phase_c.py <tag>`: fetch a full job's France p → bge re-blend (P13 + P19 + P26 r) → profile (t_eq_d4, P6 t, t-curve,
    sibling / hn diagnostics, churn vs #9ab) → `prob_test_France_<tag>_bge.parquet`. 20 s, < 1 GB.
  - `bge_topup.py build <tags>`: P19's build for band pairs without r, then P19's `launch` with `--prefix amc26-p26bge`.
  - `rh_build.py`: tiny rehearsal channels.
  - `loco_table.py`, `log_loco.py`: tables and ledger rows from `data/work/p26/res/*.json`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences / data:** LightGBM (MIT), scikit-learn (BSD); the bge CE is P13's (Apache-2.0). No external data.
  France uses unlabelled test features only (IDF world, candidate lists, house numbers), which organiser ruling 2 allows.
- **France is not LOCO.** Two proxies gain +1.8 to +2.0 pt, survive the CE, and hold under every blend weight tried. France's own gain is unmeasured until the LB.
  - Label-free signs agree: 3× fewer sibling-offset accepts at equal matches/S1.
  - Only 9.9% of France S1 change vs #9ab, and the profile stays in P6's bands.
- **The support features depend on candidate-list density.** France has 15.6 unpruned candidates per S1, train 14.2. Both are computed on the full list, as in training.
- **The France threshold is still set by D4's empty rate.** In LOCO, P6's rule tracks the oracle, but France's LB rejected the P6 rule for m2 (0.78 < 0.95).
- **AWS spend ≈ $1.95, 16 jobs, none left running:**
  - 11 on ml.m6i.4xlarge ($0.97/h): about 2,900 s, $0.78;
  - 3 on m8i-flex.4xlarge + 1 on r8i-flex.4xlarge: 2,977 s, ≈ $1.0 (not in the Pricing API, so estimated);
  - 1 on ml.g5.xlarge: 259 s, $0.12.

## Open questions   (for the master; each with your best guess)
- **France t for the hn file: 0.986 (D4's rate) or looser?** Best guess: 0.986 first. If it beats #9ab, spend the next slot on
  `sub26_hn_t975`: siblings are 3× rarer now, so the recall the high t sacrificed may be worth more than it was for m2.
- **Combine with P20's R1 (French suffix fillers) or P15's sibling rule?**
  - The sibling rule is nearly redundant: about 0.4 accepted sibling-offset pairs per 1k remain.
  - R1 changes cores, so France X must be rebuilt in P20's world and c_hn scored there: one `full`-style run if R1 wins on the LB.
- **Train c_hn on folds 2–9 for France?** Not done. 4× data gave m2 about +0.0004 in-country (P5). Best guess: small, not worth a slot.
