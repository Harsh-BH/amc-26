# P45-ens-stages — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- DRAFT 3 (03:25 IST). **One member passes, and it is not a tree:** our shipped bge CE scored on EVERY argmax
  pair (`r_all`, not only the 0.02<p<0.98 band), as an a10 column. Cross-fit: **+0.000285 ± 0.000032 LB-eq over #34
  (9 SE)**, and +0.000325 ± 0.000032 on top of P44's two group columns. Both halves and both countries are positive,
  the leakage check is clean, and the gain is the same on "clean" S1.
- Everything else is ≈ 0: the pruner zoo (A), 19 tree/NN matcher members (B, ≤ +0.00008 cross-fit), and a learned
  bge+Qwen combiner (C, = Qwen). Tree members' errors correlate 0.94–0.99 with m2_big; r_all's correlate 0.35–0.48.
- sub45 = output/sub46_on43 with India/US from a10 + [sx_psum, n_same_src_acc, l_r_all]; holdout h0→h1 +0.000533 ±
  0.000060 over #34. Test r_all is being scored now (3 GPU jobs, ETA 03:50); then build, profile, validator.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Pending the test build. Plan: `te45.py rall → probs → combine output/sub46_on43 output/sub45`, validator
  `--check-ids` on SageMaker (P43's job), plus a SUBMISSIONS.md row.
- Pipeline change for P9 / P43:
  - Score bge `ce_bge_pool.pt` on every #8c argmax pair of India/US. Test: 5.64M pairs outside the band (the band
    already has r).
  - Add three columns to a10: `sx_psum`, `n_same_src_acc` (P44 ctx.add3) and `l_r_all` = logit of r on the argmax pair.
  - Booster: `data/work/p45/a10r_h0.txt`; column order in `a10r_cols.json`; a10's params; guard; t 0.71; argmax kept from #8c.

## Evidence         (table: ledger name | world | key metrics | verdict)
### B′. Full-coverage CE and P44 columns, stacked into a10 (cross-fit = mean of fit-h0→report-h1 and the swap)
| arm | base | h0→h1 LB-eq | h1→h0 | cross-fit | India / US (cross-fit) | verdict |
|---|---|---|---|---|---|---|
| + l_r_all | #34 | +0.000326 ± 0.000047 | +0.000244 ± 0.000042 | **+0.000285 ± 0.000032 (9.0 SE)** | +0.000395 / +0.000261 | **PASS** |
| + P44 sx_psum, n_same_src_acc | #34 | +0.000145 ± 0.000039 | +0.000141 ± 0.000039 | +0.000143 ± 0.000028 | +0.000145 / +0.000199 | (P44's) |
| + l_r_all | #34 + P44 cols | +0.000387 ± 0.000050 | +0.000263 ± 0.000041 | **+0.000325 ± 0.000032 (10.1 SE)** | +0.000439 / +0.000313 | **PASS** |
| + r_all + 16 zoo members + zoo stats | #34 + P44 cols | +0.000430 | +0.000318 | +0.000374 ± 0.000036 | +0.000530 / +0.000329 | pass, but the zoo adds only ≈ +0.00005 over r_all |
| sub45 booster (#34 + P44 cols + l_r_all) | #34 | **+0.000533 ± 0.000060** | – | – | +0.000686 / +0.000554 | ship candidate |
| + rank_sx2 / et2 / e5 / noctx / mlp, each alone | #34 + P44 cols | | | +0.000032 / −0.000011 / +0.000024 / +0.000023 / +0.000034 | | drop (≤ 1.8 SE) |

**Leakage check (master's request):**
- **Code:** ce_bge_pool was trained on P4b trainband folds ≥ 4, with fold 3 as validation (`rerank.band` reads
  fold ≥ 3; `p13.build` keeps ≥ 4). Every holdout S1 is fold 0: 88,168 India, 132,190 US.
- **Data:** exact "name | addr" matches between holdout S1 texts and P13's actual train_/val_ channel texts:
  **0 / 0**.
- **SX overlap:** 2.7% (India) and 3.6% (US) of holdout argmax SX texts also appear in CE training, paired with
  other S1. That is the same protocol as every matcher.
- **Gain on clean vs overlap S1**, per S1:
  - India clean +0.000543 ± 0.000095 vs overlap +0.000657; swap +0.000330 vs +0.000274.
  - US clean +0.000329 ± 0.000066 vs overlap +0.000371; swap +0.000287 vs +0.000323.
  - No leak signal.

**Where the gain is** (report half, per country, rall.py):
- 50–76 vetoes of pairs with #8c p ≥ 0.98; 74–90% of them are false.
- 5–31 rescues of pairs with #8c p < 0.02; 80–100% are true.
- Band reshuffles: adds 69–79% true; drops 15–37% false.
- Flip rates on holdout h1 (sub45 vs #34), per #34-accepted pair: India adds 0.00103 / drops 0.00106; US adds 0.00130 / drops 0.00082.

**Error correlation (Platt-calibrated) with m2_big on #8c argmax pairs, India / US:**
- tree boosters: 0.95–0.99; fsub 0.95 / 0.96
- mlp 0.85 / 0.90; mlp2 0.84 / 0.89
- rank_s1 0.87 / 0.90; noctx 0.74 / 0.76; rf2 0.71 / 0.84; et2 0.46 / 0.55; rank_sx2 0.38 / 0.39
- e5 cosines: 0.31–0.37
- **r_all: 0.35 / 0.48**; outside the band 0.28 / 0.31
- a10 vs m2: 0.56 / 0.64

### A. Pruner zoo (P10 operating point; holdout; combiner τ matched on half 0, reported on half 1 and the swap)
- Setup: 10 new members (seeds, folds 4–5 / 6–7 / 8–9, extra_trees, feature_fraction 0.5, all 8 folds "wide",
  cross-entropy at lr 0.05) + P10's shipped and big pruners.
- Yardsticks:
  - **m2**: argmax over the kept pairs on m2_big p, t 0.69. It treats every arm the same way.
  - **p34**: pairs in #6's set keep their shipped final p; new pairs get m2 p. This penalises every non-shipped pruner.
- Shipped pruner: holdout size 4.29–4.30/S1; recall of blocked true pairs India .9955 / US .9958.

| combiner | equal size: recall In / US | equal size: ΔF m2 h0→h1 / swap | equal size: ΔF p34 h0→h1 / swap | equal recall: size (ship 4.29) | verdict |
|---|---|---|---|---|---|
| mean of 12 logits | .9957 / .9959 | +0.000020 ± 0.000005 / +0.000022 ± 0.000012 | +0.000000 / +0.000006 | 4.270 / 4.289 | drop (≈ 0) |
| LR stacker on 12 logits | **.9963 / .9965** | +0.000022 ± 0.000007 / +0.000009 ± 0.000014 | −0.000018 / −0.000021 | **4.196 / 4.211 (−2.2%)** | drop: = P10's single big pruner |
| P10 big (one 600×63 model) | .9963 / .9963 | +0.000016 / +0.000022 | −0.000019 / −0.000003 | 4.199 / 4.214 | (P10 prior art) |
| max of zoo | .9952 / .9957 | +0.000010 / +0.000023 | −0.000018 / −0.000007 | 4.312 / 4.327 | drop |
| mean of data-diverse (ship + f45/f67/f89) | .9954 / .9957 | +0.000011 / −0.000006 | −0.000032 / −0.000027 | 4.295 / 4.314 | drop |
| best single new member (xent) | .9957 / .9962 | +0.000021 / +0.000017 | −0.000007 / −0.000006 | 4.251 / 4.267 | drop |
- Pruner error correlation near the cut is 0.97–0.999.

### B. Matcher zoo stacked into a10 (vs #34 refit on the same half, guard, t 0.71; LB-eq = 0.47·India + 0.38·US)
| arm | members | h0→h1 | h1→h0 | cross-fit | verdict |
|---|---|---|---|---|---|
| #34 refit vs #8c (sanity) | – | +0.000209 ± 0.000056 | +0.000175 ± 0.000048 | | = P34 exactly |
| z_all, P22/P25 trees | 10 | +0.000020 ± 0.000028 | −0.000020 ± 0.000028 | ≈ 0 | drop |
| z_all + rank_s1, rank_sx2, rf2, et2 | 14 | +0.000057 ± 0.000039 | +0.000010 ± 0.000032 | +0.000034 | drop |
| z_all + mlp + noctx | 16 | +0.000101 ± 0.000037 | +0.000028 ± 0.000034 | +0.000064 ± 0.000025 | drop |
| z_all_stats (16 + mean/std/min/max) | 16 | +0.000106 ± 0.000037 | +0.000049 ± 0.000039 | +0.000077 ± 0.000027 (2.9 SE) | drop (< +0.00015) |
| z_nn (mlp + e5 cosines) | 4 | +0.000067 ± 0.000031 | +0.000029 ± 0.000032 | +0.000048 ± 0.000022 | drop |
| z_e5 (3 cosines) | 3 | +0.000017 | −0.000009 | +0.000004 | drop |
| lr_zoo (logistic meta on lp, lm2, logits; replaces a10) | 12 | −0.000215 vs #34 (−0.000006 vs #8c) | −0.000167 | | drop |
| lr_zoo + a10 on top | 12 | −0.000013 | +0.000012 | | drop |
- Member band logloss: every tree booster is within 0.0006 of m2_big (India 0.0571). mlp's val logloss is 0.0262
  vs m2_big's 0.0229 on fold 1; noctx 0.0381; rf2 0.103; et2 0.135.

### C. Reranker: learned bge + Qwen combiner (P38 LOCO files, P38's France procedure; France-F0.5, unseen country half 1)
| arm | US→India (mean w) | India→US (mean w) | verdict |
|---|---|---|---|
| bge (shipped CE) over no-CE | +0.00280 ± 0.00023 | +0.00148 ± 0.00015 | reference |
| Qwen over no-CE | +0.00420 ± 0.00024 | +0.00257 ± 0.00018 | P38 |
| logit mean (P38's ens) − Qwen | −0.00047 ± 0.00012 | −0.00018 ± 0.00010 | drop (P38) |
| **learned stack [logit p, logit r_bge, logit r_qwen] − Qwen** | +0.000065 ± 0.000056 | +0.000103 ± 0.000080 | **drop** (≈ +0.00001 LB-eq) |
| stack + \|Δlogit\| − Qwen | +0.000042 ± 0.000054 | +0.000065 ± 0.000086 | drop |
| stack, honest cross-fit weights − Qwen | +0.000086 ± 0.000067 | +0.000157 ± 0.000072 | drop |
- The fitted OOD weights give bge ≈ 0.01 and Qwen 0.36. The combiner switches bge off.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Pruner ensembles (A):** +0.00002 at equal size. At equal recall they are 2.2% smaller, which one bigger pruner
  (P10 big) already gets. The pruner members are near-identical (error corr ≥ 0.97).
- **Tree / NN matcher stacking (B):**
  - Best cross-fit +0.000077 ± 0.000027. Algorithm diversity (xgb, cat, goss, dart, extra) is no more diverse than seeds.
  - Only forests, rankers and MLPs lower the correlation, and they are weaker.
  - A logistic meta-learner over booster logits is ≈ #8c (+0.00001), i.e. it throws a10's context away.
- **bge + Qwen combiners (C):** the learned combiner = Qwen alone; the plain mean is worse than Qwen.
- **e5-small frozen cosines:** too weak (errors 0.31–0.37 correlated with m2_big only because they carry little
  signal); +0.00002.
- **Engineering:** the first forest / ranker runs stopped at 24–85 trees (early stopping on a flat metric); the
  reruns keep the best of 300 / 2000. The first pruner-eval run was OOM-killed by float64 logit matrices.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `zoo.py oof | extra <ce_job> <e5_job> | corr | corr2 | eval [arms]` (env `P45_BASE=cols` = the base to beat):
  - Writes `data/work/p45/oof/holdout_{India,US}.parquet` + README (P44's inputs).
  - eval takes 1–2 min and < 1.5 GB. `summ.py` prints the cross-fit summary.
- `te45.py fit | rall | probs | combine SRC OUT`: sub45. `rall.py`: flip decomposition. `leak.py`: leakage check.
- Jobs:
  - `solution_p45.py` = P25's job + `group` (lambdarank) + `drop` (feature subsets).
  - `solution_p45n.py`: MLP members.
  - `solution_p45a.py` + `launch_a.sh`: pruner zoo.
  - `solution_p45e.py`: e5 cosines.
  - `solution_p45c.py` + `launch_c.py`: full-coverage CE on test. Texts are built on the instance, then P13's
    bge_ckpt path runs.
  - `cebuild.py` / `tekeys.py`: holdout / test channels.
  - `launch.sh` + `variants.json` put hp in hyperparameters.json, never argv.
- Test r_all cost: 5.64M pairs, ~25 min on 3 GPUs (A10G ≈ 1,300–1,900 pairs/s).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- bge-reranker-v2-m3 is Apache-2.0 (already shipped); multilingual-e5-small is MIT (diagnostic only, not shipped).
  LightGBM / XGBoost / CatBoost / sklearn / PyTorch. No external data.
- France is untouched: a10 runs on India/US only, so an unseen country keeps its path.
- About 60 arms share the same halves. The passing arm is 9–10 SE with both halves ≥ 5 SE; the rest are ≤ 3 SE.
- Covariate shift: the test flip profile vs holdout h1 is pending (te45.py probes).

## Open questions   (for the master; each with your best guess)
- Pending the test build.
