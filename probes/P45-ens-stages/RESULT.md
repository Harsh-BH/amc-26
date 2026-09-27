# P45-ens-stages — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Tree ensembles add nothing at any stage.** Pruner zoo: +0.00002. 20 matcher members (trees, forests,
  lambdarank, MLPs, e5 cosines) stacked into a10: ≤ +0.00008 cross-fit, and 0 on top of the winner. A learned
  bge+Qwen combiner equals Qwen alone. Tree members' errors correlate 0.94–0.99 with m2_big.
- **The one diverse member that passes is our shipped bge CE scored on EVERY argmax pair (`r_all`), not only the
  0.02<p<0.98 band.** As an a10 column it gives cross-fit **+0.000285 ± 0.000032 LB-eq over #34 (9 SE)**. With
  P44's sx_psum + n_same_src_acc: **+0.000469 ± 0.000040 (11.8 SE)**. Leakage check clean.
- **Staged `output/sub45`** = #46_on43 with that India/US; validator PASS `--check-ids`; France byte-identical.
  **Expected LB ≈ +0.0004 over #46_on43**; test flip profile within 1.5×. Fallback: `output/sub45_psumoff`.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Upload `output/sub45`, or give P44's sub44r the same column.** Then port r_all into run.py.
- **Files:**
  - `output/sub45/matching_results.tsv`: md5 db7e4e6eeefd92a8fd9358e0ebb25d38; validator `--check-ids` PASS
    (SageMaker p45-val45-0402).
  - `output/sub45/candidate_pairs.tsv`: hardlink of #46_on43's, md5 b3ccba0d….
  - 9,969 India/US lines change (India 5,373, US 4,596). France's 259,452 lines are byte-identical to #46_on43.
- **Fallback `output/sub45_psumoff`** (md5 54eaa8c0…, PASS): without `sx_psum`, the column that reads competitor p
  and is partly in-sample on the holdout. Cross-fit +0.000295 ± 0.000034.
- **Model:**
  - Booster `data/work/p45/a10r_h0.txt`; columns in `data/work/p45/a10r_cols.json`: a10.COLS + `sx_psum`,
    `n_same_src_acc`, `l_r_all`.
  - a10's params (200 × 31, lr 0.05, min_data 200, init_score logit p), fit on India+US holdout half 0.
  - Decision: guard (hn_cls 1 keeps #8c p), t 0.71, acc = am_8c & p′ ≥ 0.71.
- **Features** (all label-free; computed identically on holdout in zoo.load and on test in te45.probs):
  - `sx_psum`: #8c p summed over every candidate S1 of the SX (P44 ctx.add3).
  - `n_same_src_acc`: other #8c-accepted argmax SX of the same S1 from the same source.
  - `l_r_all`: logit(clip(r, 1e-4, 1 − 1e-4)) of ce_bge_pool.pt on the argmax pair. Test: P13's band r where it
    exists, else the new scores.
- **run.py port (P9/P43):**
  1. Score ce_bge_pool.pt (P13 `bge_ckpt` path) on every India/US #8c argmax pair that has no band r. Test:
     5,636,451 pairs, ~40 min on 3 GPUs (A10G 1,300–1,900 pairs/s; `solution_p45c.py` builds texts on the
     instance).
  2. In `a10.apply`, add the three columns and load `a10r_h0.txt` with `a10r_cols.json`'s order.
  3. Acceptance check: the test decision reproduces `output/sub45` India/US lines (`te45.py probs` then
     `combine output/sub46_on43 OUT` gives the same md5, checked).
- **Candidate-set size:** unchanged, 4.83/S1. No pruner-ensemble change is recommended (see A).

## Evidence         (table: ledger name | world | key metrics | verdict)
### B′. Full-coverage CE + P44 columns, stacked into a10 (cross-fit = mean of fit-h0→report-h1 and the swap; LB-eq = 0.47·India + 0.38·US)
| ledger name | base | h0→h1 LB-eq | h1→h0 | cross-fit | India / US | verdict |
|---|---|---|---|---|---|---|
| B-z_r_all-vs34 | #34 | +0.000326 ± 0.000047 | +0.000244 ± 0.000042 | **+0.000285 ± 0.000032 (9.0 SE)** | +0.000395 / +0.000261 | **PASS** |
| (P44 cols) base_vs_34 | #34 | +0.000145 ± 0.000039 | +0.000141 ± 0.000039 | +0.000143 ± 0.000028 | +0.000145 / +0.000199 | P44's |
| B-z_r_all-vs34+p44ctx | #34 + P44 cols | +0.000387 ± 0.000050 | +0.000263 ± 0.000041 | **+0.000325 ± 0.000032 (10.1 SE)** | +0.000439 / +0.000313 | **PASS** |
| B-a10r-sub45-booster (#34 + P44 cols + l_r_all) | #34 | **+0.000533 ± 0.000060** | +0.000405 | **+0.000469 ± 0.000040 (11.8 SE)** | +0.000584 / +0.000512 | **ship: sub45** |
| sub45_psumoff booster (n_same_src_acc + l_r_all) | #34 | +0.000368 ± 0.000047 | +0.000222 | +0.000295 ± 0.000034 (8.6 SE) | +0.000359 / +0.000332 | fallback |
| every zoo member / all 20 + stats, on top of sub45's columns | sub45 base | −0.000026 … +0.000007 | −0.000018 … +0.000040 | −0.000012 … +0.000018 (≤ 0.7 SE) | | drop |
| B-r_all-leakage-check | – | 0 holdout S1 in CE folds or texts; clean-S1 gain = overlap-S1 gain (below) | | | | clean |
| stage-sub45 / stage-sub45_psumoff | test | validator PASS, profile below | | | | keep |

**Leakage check (master's mandatory request):**
- **Code:** ce_bge_pool trained on P4b trainband folds ≥ 4, validated on fold 3 (`rerank.band` reads fold ≥ 3;
  `p13.build` keeps ≥ 4). Every holdout S1 is fold 0 (88,168 India / 132,190 US).
- **Data:** exact "name | addr" overlap between holdout S1 and P13's real train_/val_ channel texts: **0 / 0**.
- **SX overlap:** 2.7% / 3.6% of holdout argmax SX texts also appear in CE training, paired with other S1. That is
  the protocol every matcher shares.
- **Per-S1 gain, clean vs overlap:**
  - India +0.000543 ± 0.000095 vs +0.000657 (swap +0.000330 vs +0.000274).
  - US +0.000329 ± 0.000066 vs +0.000371 (swap +0.000287 vs +0.000323).

**Where the gain is** (report half, per country):
- 50–76 vetoes of #8c p ≥ 0.98 pairs; 74–90% of them are false.
- 5–31 rescues of p < 0.02 pairs; 80–100% are true.
- Band reshuffles: adds 69–79% true; drops 15–37% false.
- The same veto signal P30 found for France (a full-coverage bge veto, already in P31's France rules), here learned
  inside a10 for India/US.

**Test profile (label-free; per #34-accepted pair, test / holdout h1):**
- sub45: adds 0.75× India / 0.71× US; drops 1.15× / 1.45×.
- Expected precision of the test flips, from holdout strata (region × l_r_all × p′):
  - adds 0.87 India / 0.81 US (holdout 0.86 / 0.80);
  - vetoes 0.70 / 0.71 false (holdout 0.57 / 0.60).
- The extra test vetoes sit in the logit(r) < −4 bin (63% / 55% of test vetoes vs 42% / 29% on holdout).
- Break-evens are about 0.67–0.75 for adds and about 0.33 for drops, so both clear them.
- Shape: India empty 5.776% → 5.782%, 3.3813 → 3.3798 matches/S1; US 5.786% → 5.789%, 3.3908 → 3.3899.
- sub45_psumoff: adds 1.04× / 1.03×, drops 1.20× / **1.70×**; expected veto false share 0.73 / 0.69.

**Calibrated error correlation with m2_big on #8c argmax pairs, India / US:**
- tree boosters: 0.95–0.99 (fsub 0.95 / 0.96)
- mlp 0.85 / 0.90; mlp2 0.84 / 0.89; rank_s1 0.87 / 0.90; noctx 0.74 / 0.76; rf2 0.71 / 0.84; et2 0.46 / 0.55;
  rank_sx2 0.38 / 0.39
- e5 cosines 0.31–0.37
- **r_all 0.35 / 0.48**; outside the band 0.28 / 0.31
- a10 vs m2: 0.56 / 0.64

### A. Pruner zoo (P10 operating point; holdout; combiner τ matched on half 0, reported on half 1 and the swap)
- Setup: 10 new members (seeds, folds 4–5 / 6–7 / 8–9, extra_trees, feature_fraction 0.5, all 8 folds "wide",
  cross-entropy at lr 0.05) + P10's shipped and big pruners.
- Yardsticks:
  - **m2**: argmax over the kept pairs on m2_big p, t 0.69. It treats every arm the same way.
  - **p34**: pairs in #6's set keep their shipped final p; new pairs get m2 p. This penalises every non-shipped pruner.
- Shipped pruner: holdout size 4.29–4.30/S1; blocked-true recall .9955 / .9958.

| ledger name | equal size: recall In / US | equal size: ΔF m2 h0→h1 / swap | equal size: ΔF p34 h0→h1 / swap | equal recall: size (ship 4.29) | verdict |
|---|---|---|---|---|---|
| A-mean_all (12 logits) | .9957 / .9959 | +0.000020 ± 0.000005 / +0.000022 ± 0.000012 | +0.000000 / +0.000006 | 4.270 / 4.289 | drop (≈ 0) |
| A-lr (LR stacker, 12 logits) | **.9963 / .9965** | +0.000022 ± 0.000007 / +0.000009 ± 0.000014 | −0.000018 / −0.000021 | **4.196 / 4.211 (−2.2%)** | drop: = P10's single big pruner |
| A-big (P10's 600×63, one model) | .9963 / .9963 | +0.000016 / +0.000022 | −0.000019 / −0.000003 | 4.199 / 4.214 | prior art |
| A-max_zoo | .9952 / .9957 | +0.000010 / +0.000023 | −0.000018 / −0.000007 | 4.312 / 4.327 | drop |
| A-mean_data (ship + f45/f67/f89) | .9954 / .9957 | +0.000011 / −0.000006 | −0.000032 / −0.000027 | 4.295 / 4.314 | drop |
| A-xent (best single new member) | .9957 / .9962 | +0.000021 / +0.000017 | −0.000007 / −0.000006 | 4.251 / 4.267 | drop |
- Pruner error correlation near the cut is 0.97–0.999, so the members are near-identical.

### B. Matcher zoo stacked into a10 (vs #34 refit on the same half, guard, t 0.71)
| ledger name | members | h0→h1 | h1→h0 | cross-fit | verdict |
|---|---|---|---|---|---|
| (sanity) #34 refit vs #8c | – | +0.000209 ± 0.000056 | +0.000175 ± 0.000048 | | = P34 exactly |
| B-z_all-vs34, P22/P25 trees (bag1–4, goss, deep, shallow, extra, xgb, cat) | 10 | +0.000020 ± 0.000028 | −0.000020 ± 0.000028 | 0.000000 | drop |
| + rank_s1, rank_sx2, rf2, et2 | 14 | +0.000057 ± 0.000039 | +0.000010 ± 0.000032 | +0.000034 | drop |
| + mlp, noctx | 16 | +0.000101 ± 0.000037 | +0.000028 ± 0.000034 | +0.000064 ± 0.000025 | drop |
| B-z_all_stats-vs34 (16 + mean/std/min/max/mean−m2) | 16 | +0.000106 ± 0.000037 | +0.000049 ± 0.000039 | +0.000077 ± 0.000027 (2.9 SE) | drop (< +0.00015) |
| B-z_nn-vs34 (mlp + 3 e5 cosines) | 4 | +0.000067 ± 0.000031 | +0.000029 ± 0.000032 | +0.000048 ± 0.000022 | drop |
| B-z_e5-vs34 (frozen e5-small cosines) | 3 | +0.000017 | −0.000009 | +0.000004 | drop |
| B-lr_zoo-vs34 (logistic meta on lp, lm2, logits; replaces a10) | 12 | −0.000215 vs #34 (−0.000006 vs #8c) | −0.000167 | | drop |
| lr_zoo + a10 on top | 12 | −0.000013 | +0.000012 | | drop |
| dart, fsub, mlp2 (on top of sub45's columns) | 1 each | −0.000026 / −0.000024 / −0.000003 | +0.000003 / +0.000007 / +0.000006 | ≈ 0 | drop |
- Members were trained on SageMaker (folds 2–5 or 6–9; early stopping on fold 1).
- Fold-1 val logloss:
  - m2_big 0.0229; fsub 0.0233; dart 0.0243; mlp 0.0262; mlp2 0.0274; noctx 0.0381; rf2 0.103; et2 0.135.
  - Rankers are uncalibrated.

### C. Reranker: learned bge + Qwen combiner (P38 LOCO files, P38's France procedure; France-F0.5, unseen country half 1)
| ledger name | US→India (mean w) | India→US (mean w) | verdict |
|---|---|---|---|
| bge (shipped) over no-CE | +0.00280 ± 0.00023 | +0.00148 ± 0.00015 | reference |
| Qwen over no-CE | +0.00420 ± 0.00024 | +0.00257 ± 0.00018 | P38 |
| logit mean (P38's ens) − Qwen | −0.00047 ± 0.00012 | −0.00018 ± 0.00010 | drop (P38) |
| **C-stack-vs-qwen-loco** [logit p, logit r_bge, logit r_qwen] − Qwen | +0.000065 ± 0.000056 | +0.000103 ± 0.000080 | **drop** (≈ +0.00001 LB-eq) |
| C-stack_d (+ \|Δlogit\|) − Qwen | +0.000042 ± 0.000054 | +0.000065 ± 0.000086 | drop |
| stack, honest cross-fit weights − Qwen | +0.000086 ± 0.000067 | +0.000157 ± 0.000072 | drop |
- The fitted out-of-domain weights give bge 0.01 and Qwen 0.36. The combiner switches bge off.
- No multi-checkpoint Qwen arm: only final LoRA r files exist, and a zero bge weight predicts nothing from weaker
  checkpoints. France stays byte-identical.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Pruner ensembles (A):** +0.00002 at equal size. At equal recall they are 2.2% smaller, which one bigger pruner
  (P10 big) already gets. Members' errors correlate ≥ 0.97.
- **Tree / forest / ranker / MLP matcher stacking (B):**
  - Best cross-fit +0.000077 ± 0.000027 over #34, and 0 on top of r_all.
  - A new algorithm (xgb, cat, goss, dart, extra-trees boosting) is no more diverse than a new seed. Only a
    different objective (lambdarank), a forest, an MLP or a dropped feature group lowers the correlation, and each
    is weaker.
  - A logistic meta-learner over booster logits ≈ #8c: it throws away a10's context.
- **Frozen e5-small cosines:** carry little signal on pruned pairs (holdout AUC 0.55–0.70 on a sample); +0.00002.
- **bge + Qwen combiner (C):** = Qwen alone; the plain mean is worse.
- **Engineering:**
  - The first forest / ranker runs stopped at 24–85 trees (early stopping on a flat metric). The reruns train all
    rounds and keep the best (~91).
  - The first pruner-eval run was OOM-killed by float64 logit matrices; the fix is a float32 matrix per country.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`te45.py fit | rall | probs | combine SRC OUT`** (env `P45_VAR=psumoff` for the fallback): sub45 end to end.
  - fit 1 min; probs 2 min at ≤ 1.5 GB; combine 1 min.
  - Writes `a10r*_h0.txt`, `a10r*_cols.json`, `prob_test_<c>_a10r*.parquet` and the flip profiles.
  - `shift45.py`: expected test-flip precision from holdout strata.
- **`zoo.py oof | extra <ce_job> <e5_job> | corr | corr2 | eval [arms]`** (env `P45_BASE=cols` = the base to beat),
  1–2 min, < 1.5 GB. `summ.py <eval json>`: cross-fit gate.
  - **P44's inputs:** `data/work/p45/oof/holdout_{India,US}.parquet` (20 members, p_r_all, e5 cosines; README) and
    `oof/r_all_test_{India,US}.parquet` (r for every test #8c argmax pair).
- **Checks:** `rall.py` (flip decomposition, clean/overlap split); `leak.py` (fold + text overlap).
- **Stage A:** `evalA.py score <job> | eval`, ~3 min, < 2 GB.
- **Stage C:** `ce.py gate`, ~2 min, 1.3 GB.
- **SageMaker jobs** (hp in hyperparameters.json, never argv):
  - `solution_p45.py`: P25's LightGBM job + lambdarank `group` + feature-subset `drop`.
  - `solution_p45n.py`: MLP members.
  - `solution_p45a.py`: pruner zoo, trained and scored on P10's holdout rows.
  - `solution_p45e.py`: e5 cosines.
  - `solution_p45c.py`: full-coverage CE; texts are built on the instance, then P13's bge_ckpt runs.
  - Launchers: `launch.sh` / `launch_a.sh` / `launch_c.py`, with `variants.json` / `members_a.json`.
  - Channel builders: `cebuild.py` / `tekeys.py`. `fetch.sh`: member preds.
- **AWS:** $7.93 in total (on-demand list prices) for 27 jobs.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / rules:** bge-reranker-v2-m3 is Apache-2.0 and already shipped. e5-small (MIT) is diagnostic only.
  LightGBM, sklearn and PyTorch only. No external data. Nothing new ships except one extra use of the existing CE.
- **France / unseen countries:** untouched; a10 runs only for India/US. France's own full-coverage bge veto is
  P30's, already in P31's France rules.
- **sx_psum is partly in-sample on the holdout** (competitor rows carry m2's in-fold p). That is why
  `sub45_psumoff` exists. r_all enters only on the fold-0 argmax pair, so it is clean.
- **Shift:** test vetoes run 1.15–1.45× holdout's rate, concentrated where the CE is most confident. Their
  estimated precision rests on holdout strata transferring to test.
- **Multiple comparisons:** about 70 arms share the same halves. The passing arm is 9–12 SE with both halves ≥ 5 SE;
  every other arm is ≤ 3 SE.
- **Test cost:** r_all adds ~5.6M CE pairs at inference (~40 min on 3 A10G-class GPUs, ~2 h on one) to the
  reproducibility story; document it in the approach doc.

## Open questions   (for the master; each with your best guess)
- **sub45 or P44's sub44r?** Whichever has the higher holdout number with l_r_all inside. P44's extra features gave
  +0.000306 without r_all; sub45 gives +0.000533 (h0→h1). If sub44r lands at ≥ +0.0006, use it; otherwise upload
  sub45.
- **psumoff?** No. sub45 is the one within 1.5× on every profile ratio. psumoff's US vetoes run at 1.70×.
- **Fit on both halves for shipping?** No. The h0 fit is the one the h1 report validates (P34's rule).
- **Refit a10r with a larger CE (Qwen) for India/US?** It might add more (Qwen beat bge out of domain), but only
  band r exist for Qwen and a 5.6M-pair Qwen pass is ~3× bge's cost. Not worth it before the deadline.
