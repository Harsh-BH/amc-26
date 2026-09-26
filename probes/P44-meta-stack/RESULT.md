# P44-meta-stack — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The report's idea fails on our holdout. Our own combiner passes the gate in both directions.** Their logistic meta-learner
  is −0.0055 LB-eq vs #34, and their margin rule / risk buckets / SIB3 filter are ≤ +0.00002 or negative. Their "m2 alone → meta"
  jump is +0.0000 (India) / +0.0009 (US) here, against the +0.0021 they claim.
- **Ours:** a 5-seed bagged a10-style residual over #34's argmax pairs. It adds exclusivity and group context (**sx_psum** =
  #8c p-mass over the SX's candidate S1; **n_same_src_acc** = other accepted SX of the S1 from the same source) plus two new
  GBDTs' logits. Holdout LB-eq **+0.000306 ± 0.000048 (6.4 SE)** h0→h1 and **+0.000210 ± 0.000047 (4.5 SE)** on the swap,
  with India and US positive both ways. Label-free test flip profile PASS (all ratios ≤ 1.09).
- **Staged:** `output/sub44` = sub46_on43 with India/US replaced (6,453 lines; France and candidates byte-identical);
  `output/sub44_psumoff` = conservative, without sx_psum. Validator `--check-ids` on SageMaker: see Evidence.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt `output/sub44` as the India/US half** (its expected LB gain is in Risks). Keep `output/sub44_psumoff` as the fallback.
  sub44r (+ P45's r_all) is pending the test r_all file.
- Model: `data/work/p44ms/ship_s{0..4}.txt`, five LightGBM residual boosters.
  - Params: a10's (200 rounds × 31 leaves, lr 0.05, min_data_in_leaf 200, λ2 1) + bagging 0.8 / freq 1 / feature 0.8, seeds 0–4,
    deterministic, init_score = logit #8c p.
  - Fit on holdout half 0, India + US pooled, no country column.
  - Output p'' = σ(mean over boosters of logit p_s), where p_s = σ(raw + logit p) and a sibling-offset pair (hn class 1) keeps
    #8c's p (P34's guard).
  - Decision: acc = am_8c & p'' ≥ 0.71.
- Columns (`te44.FINAL`, 45):
  - a10.COLS;
  - l_m2hn and l_hn21 (logits of the two P44 GBDTs);
  - hn_d, hn_cls, hn_lev, hn_lendiff, has_hn, exact_hn;
  - lq (pruner logit), tw_any;
  - CTX2: sx_n_ge_t, sx_n_ge_05, s1_ncand, s1_rank, s1_best_nonam, n_same_hn_acc, n_acc_hn_eq_s1;
  - CTX3: s1_gap_next, s1_gap_prev, sx_psum, s1_psum, is_s3, n_same_src_acc.
- run.py port (P9): after a10_stage, for India/US:
  1. build the frame with `te44.build`'s steps: ctx.ctx2 / ctx.add3 on #8c's rows; solution_p44.hn_feats; the pruner q
     lookup; the two GBDTs' p;
  2. score with `te44.combine_p(boosters, X)`;
  3. keep #8c's am.
  The two GBDTs need the hn tables (`hn_tables.py`) and their boosters (`s3://…/amc26-p44/results/p44-{m2hn,hn21}-0927-0215/`).

## Evidence         (table: ledger name | world | key metrics | verdict)
Yardstick: #34 = #8c + P34's guarded a10, out of fold (P36's base.py; reproduced exactly: India h1 0.990200, US h1 0.989601).
Deltas are per-S1 F0.5 vs #34 on the report half, with paired SE. LB-eq = 0.468 ΔIndia + 0.383 ΔUS. Fit h0 → report h1 / swap.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| armc-asis_all(margin,nohn,nonuniq_tw,sib3) | holdout | report's decision layer on #34: −0.000063 ± 0.000027 / −0.000022 | drop |
| armc-asis_margin | holdout | margin rule (Δ < 0.10 → t ≥ 0.85), t fixed: +0.000010 ± 0.000006 / +0.000013 ± 0.000011 | drop |
| armc-tuned_margin / asis_nohn / asis_nonuniq_fq | holdout | +0.000021 / −0.000061 / −0.000019 (h0→h1) | drop |
| jump-m2alone-to-report-meta-hn21 | holdout (all rows, argmax per score) | m2 alone India 0.98272 / US 0.98366. Report meta − m2: India −0.00053 / −0.00008, US +0.00093 / +0.00085. (#8c − m2: +0.0073 / +0.0056.) Meta vs #34: −0.008 / −0.005 | drop |
| a_report_hn21 | holdout | LR [p_m2, p_hn21, \|diff\|, min, max, has_hn, exact_hn]: **−0.005541** / −0.006012 | drop |
| a_report+r+p8_hn21 / _m2hn | holdout | the same + CE r + #8c p: −0.000263 / −0.000222 | drop |
| b_a10+report_hn21_t071 / _m2hn | holdout | a10 + report columns: +0.000026 ± 0.000031 / −0.000008; m2hn: +0.000056 / +0.000054 | drop |
| o_a10+l_m2hn_t071 | holdout | a10 + m2hn logit only: +0.000032 / +0.000021 | drop |
| o_mono_gbdt, o_logit_inter, o_stack2_on_pa, o_iso_cat_pa | holdout | all within −0.00015 … +0.00001 | drop |
| o_a10+all+ctx_t071 | holdout | + CTX2: +0.000155 ± 0.000044 (3.6 SE) / +0.000082 ± 0.000043 | inconclusive |
| best_bag5 | holdout | the same, bagged: +0.000144 ± 0.000037 / +0.000126 ± 0.000043 | inconclusive |
| ctx3_seed0 / ctx3_bag5 | holdout | + CTX3: +0.000298 ± 0.000052 (5.8 SE) / +0.000182 (3.5 SE); bag +0.000292 (5.9 SE) / +0.000213 (4.5 SE) | keep |
| only+sx_psum / only+n_same_src_acc | holdout | each alone on the pre-CTX3 set: +0.000256 / +0.000206; +0.000207 / +0.000174 | keep |
| insample-check-sxpsum-oofonly | holdout | sx_psum rebuilt from out-of-fold rows only (holdout + fold-1 S1): +0.000167 ± 0.000041 (4.1 SE) / +0.000161 ± 0.000049 (3.3 SE). h1 gain from pairs with no other-S1 mass: India +0.000223, US +0.000269; from in-sample competitor mass: +0.000086 / +0.000132 | keep |
| swap-m2hn-under-ce-blend | holdout (all rows) | m2hn in place of m2_big under the bge blend (w refit): vs #8c India −0.000011, US +0.000058 ± 0.0001 (h0→h1) | drop |
| **ship_bag5** (= `te44.FINAL`, no lfreq) | holdout | **+0.000306 ± 0.000048 (6.4 SE)**: India +0.000285 ± 0.000076, US +0.000452 ± 0.000083. **Swap +0.000210 ± 0.000047 (4.5 SE)**: India +0.000229, US +0.000268 | **keep (gate)** |
| ship_psumoff (bag5, no sx_psum) | holdout | +0.000178 ± 0.000039 (4.5 SE) h0→h1 | fallback |
| test profile (sub44) | test | test/holdout-h1 ratios. India: adds/acc 0.69, adds/unacc 0.36, drops/acc 0.72, S1 changed 0.70, flips/pair 0.62 (no-mass 0.66, mass 0.72). US: 0.73 / 0.45 / 0.86 / 0.77 / 0.71 (0.92, 1.09) | **PASS** |
| test profile (psumoff) | test | max ratio 1.18 (US no-mass flips) | PASS |

2nd GBDTs (SageMaker, folds 2–5, early stopping on fold 1's 25% sample, never fold 0):

| model | features | trees | fold-1 val logloss | am-pair logloss India / US (m2_big 0.0447 / 0.0522) |
|---|---|---|---|---|
| hn21 (the report's version_anu, rebuilt) | 14 X cols + 7 hn-corruption feats | 500 | 0.0705 | 0.2205 / 0.1370 |
| m2hn | m2_big's 65 + the same 7 | 3,587 | **0.01931** (m2_big 0.02285) | 0.0415 / 0.0348 |

## What failed      (approaches tried and dropped, with the number that killed them)
- **The report's recipe.** The logistic meta on [p_m2, p_hn, …] is −0.0055 LB-eq vs #34; it cannot see the CE. With CE r and
  #8c p added it is still −0.00026 at −4.5 SE. As a10 columns: +0.00003 / −0.00001.
- **The report's decision rules** on #34 (margin, no-hn 0.80, non-unique-name 0.78, SIB3): each rule ≤ +0.00002, and all
  together −0.00006.
- **Their jump.** The 0.9903 → 0.9924 was measured on a sparse 30k-S1 dev world with pooled P/R. On our fold-0 holdout,
  m2 alone is 0.983. The master's review of the branch also found cut-offs swept on scored rows and a model trained on S1
  that overlap our holdout.
- **m2hn instead of m2_big under the CE:** ≈ #8c. The CE absorbs m2hn's pre-CE lead (+0.0006–0.0011 India, +0.0031 US).
- **Other combiner families** (monotone GBDT, logistic with interactions, isotonic per category / hn class, a 2-level stack on
  p'): all ≤ +0.00001 or negative. A per-country t tuned on the fit half costs 0.00004–0.00006 (control_34_pa_tuned: −0.000060).
  0.71 stays.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
Run from the repo root with `probes/P44-meta-stack/py.sh` (a 2.2 GB cap). Work files go to `data/work/p44ms/`.
- `hn_tables.py`: hn per q1 / qx code → `hn/{split}_{c}_{s1|sx}.parquet` (40 MB, 10 s).
- `solution_p44.py` + `config.json` + `launch.sh` / `status.sh`: the SageMaker job. Only the variant name goes through
  hyperparameters.json. `hn_feats(h1, hx)` is vectorised and asserted equal to `stack.hn_cls` on 200k real pairs.
  - hn21: 3 min on ml.m5.4xlarge. m2hn: 18 min on ml.m6i.4xlarge (5.5 GB).
- `ev.py`: the holdout frame A_{c} and the metric helpers (`per_s1`, `delta`, `lbeq`); it asserts #34 reproduces.
- `ctx.py`: `ctx2(A, d, h1, hx)` and `add3(A, d, s3)`, the one context builder for holdout and test.
- `arm_c.py`, `meta.py`, `jump.py`, `swap.py`, `best.py`, `final.py`, `abl3.py`, `diag_insample.py`, `rall.py`: the arms above.
- `te44.py fit | build c | probs c | profile | combine SRC OUT`: the ship path.
  - `P44_VARIANT=psumoff` selects the fallback.
  - fit: 1 min.
  - build: 3–4 min and ≤ 2 GB per country.
  - probs: 1–2 min.
  - combine: 1 min; it asserts SRC's India/US lines equal #34's decision, and hard-links the candidate file.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / rules:** LightGBM (MIT) and rapidfuzz (MIT). No external data. All features are label-free test-side statistics.
- **Holdout competitor rows carry in-sample p** (m2_big: folds 2–5; the bge CE: folds 4–9).
  - sx_psum and a10's own p2 read them.
  - About a third of the holdout gain comes from pairs whose SX has in-sample competitor mass, and heavy competitor mass is
    1.4× (India) to 2.5× (US) commoner on the holdout than on test.
  - The out-of-fold-only rebuild still passes (+0.000167 / +0.000161), and so does the no-mass slice alone.
- **Expected LB:** test flip rates are 0.6–0.9× the holdout's, so expect **+0.0002 (range +0.00015 to +0.0003)** over #46_on43,
  inside the public/private tie band.
- **France / unseen countries:** untouched (byte-identical to sub46_on43).
- **Assumptions:**
  - The gate is read as LB-eq with P34's weights (0.468 / 0.383).
  - US half 0 holds P17-in-sample p, so the swap's US figure is indicative only.
  - lfreq was dropped before shipping, because the S1 pool size differs between train and test.
  - Worked in the main checkout (data/ lives there), not a separate worktree.
- **Selection:** about 40 arms were run. The ship model is the pre-registered CTX3 extension, bagged. Its swap direction and the
  out-of-fold rebuild are independent confirmations.

## Open questions   (for the master; each with your best guess)
- sub44r (+ P45's r_all): pending the test file. Best guess: an extra +0.0003, if the test r_all matches the holdout's scale.
- Upload `sub46on43_recipe`-based files instead? `te44.py combine output/sub46on43_recipe output/<out>` works the same
  (≈ 1 min), provided that base's India/US lines are #34's (the combine asserts it).
