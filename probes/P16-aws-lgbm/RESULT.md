# P16-aws-lgbm — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- m2_big's recipe retrained on train folds **2–9** (25.1M pairs, 2× m2_big) in one SageMaker CPU job ($0.63): 6,683 trees, fold-1 val logloss 0.02189 vs 0.02285.
- On the #7 yardstick (report half 1, w and t refit on half 0) it scores **+0.00014 ± 0.00006 (2.6 SE)**. With #7's shipped w/t and nothing tuned, all 220k holdout S1: **+0.00014 ± 0.00004 (3.8 SE)**.
- The gain is real but below the adopt bar (≥ +0.0002 at ≥ 3 SE). **Do not adopt; keep m2_big.** The booster is kept for a possible master overrule.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Nothing.** `model.BOOSTER` stays `data/work/p4/m2_big.txt` and `model.FOLDS` stays (2, 3, 4, 5).
- **If the master overrules** (as with m2_big at +0.0004), the change is the booster alone:
  - Booster: `data/work/p16/m2_p16.txt`, sha1 tag `0726cb9a`. Also at `s3://sagemaker-114872709824-ap-south-1/amc26-p16/results/p16-lgbm-0926-0227/m2_p16.txt`.
  - Set `model.FOLDS = (2, 3, 4, 5, 6, 7, 8, 9)`.
  - **Pruner:** no refit needed. `prune.FEATS` are 30 X columns (blocking signals + similarities), trained on folds 2–3; it never sees M2's p. The kept holdout pairs are identical: 2,783,306 for both boosters.
  - **CE:** no retrain. It learns from a fold-2 LightGBM band (`rerank.band`), not from m2_big.
    - The inference band shifts by 1.6% of its pairs (46,198 new holdout band pairs, now CE-scored and cached in `data/work/p9/r_holdout_*_bb0c29f0.parquet`).
  - **Blend w:** the half-0 refit moves to **[0.6433, 0.6567, 0.263]** (m2_big: [0.6236, 0.6702, 0.2674]), i.e. more weight on M2 and less on the CE. The half-0 t is 0.65 (m2_big 0.63).
    - #7's shipped w [0.617, 0.666, 0.261] with t 0.71 gives the same gain (+0.00013 on half 1), so the refit is optional.
  - **Test scores:** not produced, because step 4 was conditional on adoption.
    - `run.py --split test --stage model` with `M.BOOSTER` pointed at m2_p16 builds them itself: the p9 cache is keyed by the booster hash, and missing CE band scores are filled under the GPU lock.
    - Or: `.venv/bin/python probes/P16-aws-lgbm/p16.py test` → `data/work/p16/prob_test_<c>.parquet` (s1, sx, q1, qx, p; pruned; India/US band CE-cached).
    - Expect ~1.4× m2_big's M2 scoring time (6,683 vs 4,930 trees).
  - **France:** t = 0.95 was LB-tuned on m2_big's p. Check the new booster's France profile first: `p16.py france` prints the empty-S1 rate and matches/S1 at t 0.69 / 0.78 / 0.88 / 0.95. m2_big's reference: 0.0534 / 0.0569 / 0.0620 / 0.0685, the same as P14's profile-sub6.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| (check) | train | `solution_p16.load` == `train_matcher.load` bit for bit (float16 folds 2–9 and float32 fold-1 val, two real X files, 30% / 25% S1 samples); file order = train_matcher's | keep |
| (upload) | aws | 19 train X files, 1.226 GB → s3://sagemaker-114872709824-ap-south-1/amc26-p16/data/v1/p4/X/ in 3 min 52 s | — |
| rehearse-1pct | train-1pct-slice | sm_train.py → solution.py with SageMaker's env and hp encoding, on a 1% S1 slice: 252,675 train / 9,132 val pairs, early stop at 450 trees. S3 checkpoint, status and booster written; the booster loads and scores through `model.predict` (AUC 0.9996 on a slice file). Deadline path: deadline 0 → stops after round 1 and still saves | keep |
| sm-job-launch | aws | `p16-lgbm-0926-0227`, ml.m7i.4xlarge on-demand ($1.018/h, Pricing API), MaxRuntime 4 h (worst case $4.07). In the job: 25,123,629 train pairs (24.1% true), 793,402 val pairs (= m2_big's), load 29 s | — |
| M2-p16-train | train | Early-stopped at **6,683 trees** (m2_big 4,930). Fold-1 val logloss **0.021886** vs m2_big 0.02285 (−4.2%). 35 min in the job (0.56 s/round at first, 0.29 s/round later), peak 8.4 GB. Billable 2,215 s = **$0.63** | keep (pair level) |
| (yardstick check) | holdout-dense | My evaluator on m2_big + #7 (CE p11/ce_s2P.pt, w [0.617, 0.666, 0.261], t 0.71, pruner 0.03): all 0.98881 / half 0 0.98894 / half 1 0.98869. These equal the #7 pipe row (0.9888 / 0.9889 / 0.9887). The half-0 refit w [0.6236, 0.6702, 0.2674] = the pipe's `blend_w_refit_half0` | — |
| eval-m2_p16-vs-m2_big-7yardstick | holdout-dense-half1 | w and t refit on half 0 for each booster. **0.988814 vs 0.988670: +0.000144 ± 0.000055 (2.6 SE)**, 335 S1 up / 287 down. India 0.98980 vs 0.98964, US 0.98816 vs 0.98802 | **drop** (< +0.0002, < 3 SE) |
| (same row) | holdout-dense-all | #7's shipped w/t, nothing tuned on the holdout: 0.988953 vs 0.988813, **+0.000139 ± 0.000037 (3.8 SE)** over all 220,358 S1 | real, below bar |
| (same row) | holdout-dense-half1 | M2 alone, no CE (t on half 0: 0.72 vs 0.71): 0.984107 vs 0.983289, **+0.00082**. About 17% of it survives the CE, the same shape as P4 (4 folds vs 1: +0.0020 without the CE, +0.0004 with it) | — |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Adopting folds 2–9.** It gains +0.00014 after the CE, below the +0.0002 bar.
  - Why the gain is small: the extra data mostly fixes pairs in the uncertain 0.02–0.98 band, which the cross-encoder already rescores.
  - Doubling the data again (folds 2–5 → 2–9) buys about a third of the previous step's +0.0004.
- **A mid-run checkpoint preview** (round 3000) of the evaluation was stopped when the job finished; it has no result.
- **Laptop cost of the evaluation.** Scoring the new booster's 2.78M pruned holdout pairs took ~70 min on the loaded laptop (load average 12–24). CE-scoring the 46k new band pairs took 13 min (49–68 pairs/s under contention). The cached rerun of the eval takes ~3 min.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `probes/P16-aws-lgbm/solution_p16.py`: the SageMaker job, shipped as solution.py next to sm_train.py.
  - `main(train_dir, test_dir, model_dir, out_dir, params)`: train_matcher.main's recipe on pyarrow. Params come in as hp (`params` = model.PARAMS, `folds`, `rounds`, `es`, `val_frac`, `deadline_s`, `ckpt_every`, `s3`).
  - Writes `model_dir/m2_p16.txt` + `result.json`. Also uploads the booster, `status.json` and checkpoints every 1000 rounds to `s3`.
  - Stops at `deadline_s` like an early stop (the best iteration is kept).
  - Full scale: 25.1M pairs, 35 min on ml.m7i.4xlarge, 8.4 GB.
- `probes/P16-aws-lgbm/p16.py` subcommands:
  - `check`: loader equivalence + file order, ~2 min, 1.1 GB.
  - `tiny`: 1% slice.
  - `rehearse`: the job on the slice, ~12 min on the loaded laptop.
  - `launch [dry]`, `status`, `fetch [ckpt]`.
  - `eval [booster]`: the #7 yardstick vs m2_big. Cold ~95 min, cached ~3 min; 1.0 GB, plus 1.4 GB in the CE process.
  - `france [booster]`: France test profile.
  - `test [booster]`: pruned test scores → `data/work/p16/prob_test_<c>.parquet`.
  - `france` and `test` were not run end to end for m2_p16 (not adopted). `profile()` itself was checked on m2_big against P14.
- `probes/P16-aws-lgbm/requirements_sm.txt`: lightgbm==4.7.0, pyarrow==25.0.1 (the CPU DLC has numpy / scipy / boto3).
- Artefacts:
  - `data/work/p16/m2_p16.txt` (94 MB) and `data/work/p16/eval_final.{log,json}`.
  - `data/work/p9/m2_holdout_{India,US}_0726cb9a_6a23c2ec_0.03.parquet`: the pipeline's cache format, so `run.py --split holdout` with this booster starts warm.
  - Extra rows in #7's CE caches `data/work/p9/r_holdout_{India,US}_bb0c29f0.parquet`: 46,198 new pairs, keyed by pair, so they are harmless to other boosters.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences / data:** LightGBM (MIT) and pyarrow (Apache-2.0), as in the pipeline. The job reads only our own train X files; there is no external data or API.
- **Rehearsal scope:** the rehearsal ran sm_train.py → solution.py in the laptop venv, not in the DLC container (not pulled; ~0.2 MB/s downlink). The real job then ran in the DLC without errors.
- **Assumption (bar):** the bar is read as both ≥ +0.0002 and ≥ 3 SE. The half-1 figure fails both; the untuned all-S1 figure passes 3 SE but fails the magnitude.
- **France / unseen countries:** France t 0.95 was LB-tuned on m2_big's p. A booster with more trees may be calibrated differently on France, which is why the France profile must be checked before any overrule.
- **Cost:**
  - SageMaker: 2,215 billable s × $1.01808/h = **$0.63** (the 30 GB volume adds < $0.01).
  - S3: amc26-p16/ holds 1.45 GB (train X copy, source, booster, model.tar.gz), ≈ $0.04/month.
  - Rehearsal S3 objects were deleted. Delete the whole prefix when done: `aws s3 rm s3://sagemaker-114872709824-ap-south-1/amc26-p16/ --recursive --profile harsh`.
  - **Total spent ≈ $0.63.**

## Open questions   (for the master; each with your best guess)
- **Overrule, as with m2_big?** Best guess: no.
  - +0.00014 is about a third of m2_big's step, and it costs 1.4× M2 scoring time on test.
  - It moves t / w and possibly France's calibration, all for a gain the size of the LB differences between the #6 variants (±0.0001–0.0002 in the pipe ledger).
- **Is more M2 data worth anything else?** Best guess: yes, where the CE doesn't already cover the pairs.
  - The no-CE gain is +0.0008, and it is concentrated in the band.
  - So the extra folds matter most for France, which is not reranked. If P13's France CE recipe is not adopted, the m2_p16 booster could be used for France only. That needs its own France-profile / LB check, and it is not measurable on labelled data here.
- **Reuse the S3 copy of the train X for other training jobs?** It is ready at `s3://…/amc26-p16/data/v1/`. `p16.py launch` takes about 2 min to adapt (new hp).
