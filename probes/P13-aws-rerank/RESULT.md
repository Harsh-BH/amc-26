# P13-aws-rerank — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **India/US: yes.** bge-reranker-v2-m3 (Apache-2.0, 568M) fine-tuned for 2 epochs on #5's CE set beats #7's shipped e5-small CE (P11 s2P).
  - Full-corpus holdout, shipped rule, blend refit on half 0: **0.98948 vs 0.98888, +0.00059 ± 0.00007** (8 SE).
  - On the D3 yardstick: +0.00064 vs s2P, +0.00099 ± 0.00008 vs #5's CE.
- **France: yes, with label-free weights.** Blend weights fit in-domain do not transfer (bge India→US −0.0016), but weights fit out-of-domain do.
  - With the real France procedure the two unseen-country proxies gain **+0.0028 ± 0.0002 (US→India) and +0.0015 ± 0.0002 (India→US)**. That procedure is the mean out-of-domain weights, the shipped rule, and P6's label-free threshold.
  - e5-small with the same recipe: +0.0011 / +0.0009.
  - Caveat: France is further out-of-domain than the proxy. The CE disagrees with the GBDT on 31% of France band pairs (proxies: 20–26%), and 15.7% of France S1 change vs #6.
- **Adopt for all three countries** via `data/work/p13/prob_test_{India,US,France}_blend.parquet`: #6's pruned candidates, t 0.71 for India/US, France 0.74 (P6's rule).
  - France stays inside P6's bands. Validator PASS.
  - If France should get a leaderboard check first, use the India/US-only fallback: France = #6's file at 0.78.
  - **Spent: $4.70** over two ml.g5.xlarge jobs: $3.87 for the training job, which failed at its end, and $0.83 for a scoring-only job.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Submission, drop-in for #6/#7.** Same rows and order as `data/work/p10/prob_test_<c>.parquet`, i.e. #6's pruned candidate set.
- Schema: `[s1, sx, p, p_gbdt]`.
- `p` = m2_big p (`p_gbdt`) blended with the bge score r on 0.02 < p_gbdt < 0.98. Outside that band p is exactly p_gbdt.
```
.venv/bin/python code/business_entity_resolution/src/submit_probs.py \
    --probs 'data/work/p13/prob_test_{country}_blend.parquet' --out output/p13_sub --t 0.71 --t-country France=0.74
```
- Validator: **PASS**. Both files have 1,732,544 rows; 99,628 matching rows are empty.
- The candidate file is #6's exact set: 130 MB, the same 22,446 S1 without candidates.
- Written to `output/p13_sub/` in this worktree.
- **Fallback, India/US only (lower risk).** The same command with France = #6's file `data/work/p10/prob_test_France.parquet` at t 0.78.
  - Use it if the master will not take the France risk below without a leaderboard check.
  - Suggested A/B, two uploads: this fallback vs the full recipe. France is ~15% of the LB.

**The blend rule.** It is P4b's: p′ = σ(a·logit p + b·logit r + c), band 0.02–0.98.
- **India, US:** w = [0.5901, 0.6319, 0.2973].
  - Fit on holdout half 0 of the *restored* checkpoint's scores (`r_bge_poolck_holdout`). These are the exact numerics the test scores were produced with.
  - t = 0.71, unchanged from #7: the shipped-rule sweep on the blended p picks 0.71.
- **France:** w = **[0.9245, 0.2595, 0.0515]**, the mean of the two out-of-domain weight vectors from LOCO. No France labels are used.
  - bge trained on US only, fit on India: [0.937, 0.278, 0.236].
  - bge trained on India only, fit on US: [0.912, 0.241, −0.133].
  - **t_France = 0.74**, from P6's `fr_diag.france_threshold` on these files with t* 0.71: France's empty-S1 rate is matched to US+India's 5.75%.
  - The same procedure on #6's files gives 0.79 (#6 shipped 0.78), so the procedure reproduces.

**The model and its scores.**
- Model: `BAAI/bge-reranker-v2-m3` + fine-tuned weights, `s3://sagemaker-114872709824-ap-south-1/amc26-p13/results/p13-rerank-0925-2257/ckpt/ce_bge_pool.pt`.
  - A local copy is at `data/work/p13/ckpt/ce_bge_pool.pt`: 624 MB, fp16, every weight except the frozen word embeddings, which come from the HF base.
  - Load with `solution_p13.Job.restore`.
- Recipe: 2 epochs over #5's 778,520 pairs (P4b trainband folds 4–9, 0.1<p<0.9 + 0.5× out-of-band), bs 128, lr 2e-5, 500 warm-up steps, linear decay, max_len 96, length buckets, bf16 autocast, fp32 master weights. 76 min on an A10G.
- Test r for **every** m2_big test band pair, so both #5-style and #6-style candidate sets can be served: `data/work/p13/r/r_bge_pool_test_{India,US,France}.parquet` (s1, sx, r) — 847,265 / 702,137 / 490,033 pairs.
- To re-blend onto another candidate file, run `p13.py testblend bge_pool_test <c> <w>`.

**Test-time cost if it goes into the pipeline.**
- A10G: 1,370–1,940 pairs/s. The pruned band (1.66M pairs) takes ~18 min; the full band (2.04M) takes 22 min.
- The laptop's RTX 3050 was not measured. From FLOPs (14× e5-small's ~2,600 pairs/s) it would be ~200 pairs/s, i.e. ~2.3 h for the pruned band.
- The SageMaker path is `solution_p13.py` phase `bge_ckpt` (rehearsed and used). Launch with:
```
p13.py launch --profile harsh --instance ml.g5.xlarge --no-spot --prefix amc26-p13 --name <job> \
  --data-s3 s3://…/amc26-p13/data/v1/ --solution probes/P13-aws-rerank/solution_p13.py --max-run 7200 \
  --hp '{"phases": ["bge_ckpt"], "ckpt": "s3://…/ckpt/ce_bge_pool.pt", "s3": "s3://…/results/<job>", "deadline_s": 6300}'
```

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| pool-bge_poolck_holdout_vs_p11s2P_plain | holdout-dense-half1 | **Shipped rule** (argmax, p ≥ t; t on half 0): #7 s2P **0.98888** (India 0.98979 / US 0.98828) → bge **0.98948** (India 0.99018 / US 0.98901), **+0.00059 ± 0.00007**, 768 S1 up / 268 down. t 0.71 both. Same with job-1's scores (pool-bge_pool_holdout_vs_p11s2P_plain) | **keep** |
| pool-bge_pool_holdout_vs_p11s2P_d3 | holdout-dense-half1 | D3 yardstick: s2P 0.98884 → bge 0.98948, **+0.00064 ± 0.00007** (927 up / 257 down) | keep |
| pool-bge_pool_holdout | holdout-dense-half1 | D3 vs #5's CE (ce5, the brief's yardstick): 0.98849 → **0.98948, +0.00099 ± 0.00008** (1,204 up / 295 down). No-CE 0.98342. Holdout band AUC r: India 0.971, US 0.956 (e5-small #5: 0.955 / 0.934 from the P13 ce5 rescore); stack 0.977 / 0.967 | keep |
| pool-bge_pool_holdout+p11:s2P_vs_p11s2P_plain | holdout-dense-half1 | logit-mean ensemble bge + s2P: 0.98930, +0.00041 ± 0.00005, i.e. **below bge alone** | drop |
| sm-job-2-score-done | aws | restored checkpoint reproduces job 1's holdout r: corr 0.999994, median \|Δlogit\| 0, p99 0.063 (fp16 rounding). Test band scored at 1,370 (India) / 1,870 (US) / 1,607 (France) pairs/s; 1,759 billable s | keep |
| loco-e5_locoUS_holdout-US>India / loco-e5_locoIndia_holdout-India>US | holdout-dense-half1 | e5-small LOCO control (P4b `ce_loco_<c>.pt`, rescored on m2_big's band); weights fit on the source country, D3 on the target's half 0: **−0.0031 ± 0.0003 / −0.0029 ± 0.0002**. Target-fit ("oracle") weights: +0.0008 / +0.0008 | drop (= P4b) |
| loco-bge_locoUS_holdout-US>India (+ _mid) | holdout-dense-half1 | bge trained on US only (483,712 pairs, 1 epoch, 367 pairs/s) → India band AUC r **0.821** (e5 0.738; p 0.931). Source-fit weights **+0.00071 ± 0.00031** (at 50%: +0.00144). Oracle weights +0.0028. In-domain +0.0047 | keep |
| loco-bge_locoIndia_holdout-India>US (+ _mid) | holdout-dense-half1 | bge trained on India only (294,656 pairs, 287 pairs/s) → US band AUC r 0.829 (e5 0.790; p 0.916). Source-fit weights **−0.0016 ± 0.0002** (mid −0.0023). Oracle +0.0014. In-domain +0.0072 | drop (in-domain w) |
| loco-xw-bge_locoUS_holdout+bge_locoIndia_holdout (+ -w mean) | holdout-dense-half1 | **Out-of-domain weights.** Each LOCO model is blended with the weights the *other* LOCO model got on *its* unseen country. D3: **India +0.00271 ± 0.00022, US +0.00133 ± 0.00015**. With the single mean vector [0.9245, 0.2595, 0.0515]: +0.00277 / +0.00142. Mid checkpoints: +0.00266 / +0.00105. e5-small: +0.00078 / +0.00075 (mean vector +0.00083 / +0.00074) | keep |
| frproxy-bge_locoUS_holdout+bge_locoIndia_holdout-w0.9245,0.2595,0.0515 | holdout-dense-half1 | **The France procedure end to end on LOCO.** Mean OOD weights on the unseen country, in-domain blend on the source; shipped rule; P6's label-free threshold (target empty rate matched to the source's at its own t*). Base = the same procedure without the CE. **US→India +0.0028 ± 0.00023** (t 0.80 → 0.71), **India→US +0.00148 ± 0.00015** (t 0.70 → 0.70) | **keep (France)** |
| frproxy-e5_locoUS_holdout+e5_locoIndia_holdout-w0.9505,0.238,0.081 | holdout-dense-half1 | same procedure, e5-small: +0.00112 ± 0.0002 / +0.00091 ± 0.00012 | keep (weaker than bge) |
| loco-bge_zs_holdout-* | holdout-dense-half1 | bge **zero-shot**: band AUC r India 0.558 / US 0.639. Blend India→US −0.00008 ± 0.00011, US→India +0.00045 ± 0.00009 | drop |
| france-threshold-prob_test_{c} (on p10) / france-threshold-prob_test_{c}_blend | test | P6 procedure, t* 0.71, ref empty 5.75%. #6's files: t_France 0.79, France 5.73% empty, 3.228 matches/S1. **P13 blend: t_France 0.74, France 5.76% empty, 3.232 matches/S1, 58.4% SX assigned — inside P6's bands** (4.5–7.5%, ≥ 2.9). US 5.74% / 3.381, India 5.76% / 3.375 (#6: 5.74 / 3.376, 5.76 / 3.374) | keep |
| france-churn-vs-6 | test | vs #6 (France m2_big p, t 0.78): **15.7% of France S1 change** (+21,604 / −23,168 pairs); 15.6% at a fixed t 0.78 (the blend), 3.5% from the threshold alone (0.78 → 0.74). CE-vs-GBDT disagreement on the band: France 31%, India 11.5%, US 10.2%, LOCO targets 20% / 26%. LOCO proxy S1 changed: 4.5% / 2.7%. Full submission `output/p13_sub`: validator **PASS** | inconclusive (risk) |
| testblend-bge-pool-on-p10 | test | India 3,969,085 rows (655,749 blended), US 3,069,192 (582,974), France 1,332,296 (424,619). Row order = #6's; p = p_gbdt outside the band (asserted). India band vs #6's p: corr 0.977, 1.7% of band pairs cross t 0.71 | keep |
| sm-job-launch / sm-job-1-failed | aws | job `p13-rerank-0925-2257`: zero-shot, both LOCOs, pooled training + holdout + checkpoint done. It then crashed at the Qwen test decision (see What failed). 8,246 billable s = **$3.87** | – |
| sm-job-2-score-launch / -done | aws | job `p13-score-0926-0216` (scoring only, user-approved): 1,759 s = **$0.83**. **Total P13 = $4.70** (at $1.69/h, the AWS Pricing API rate for APS3-Train:ml.g5.xlarge; g5 uses NVMe, so there is no EBS charge; S3 ≈ 1.2 GB, cents) | – |
| (data) | train / holdout / test | 186 MB of band texts in s3://…/amc26-p13/data/v1/ (uploaded once, 18 s). The train set is #5's CE set (P11 `data_p4b` selection). Texts are P3's, verified: #5's CE on them reproduces the cached holdout r (median \|Δr\| 0). max_len 96 truncates 0.3% of India pairs (64 would truncate 18.7%) | keep |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Job 1 died after the expensive part, on a bug I introduced.**
  - What happened: I added `"load_dtype"` to the DEFAULTS dict line and put a `# …` comment after it. That comment turned the rest of the line — `"s3": "", "qwen_test": "auto"` — into comment text.
  - Why the rehearsals missed it: all four passed both keys explicitly.
  - What was lost: the pooled model's test scoring and Qwen3. $0.83 of scoring re-run; Qwen was dropped by the master.
  - Fix: moved `load_dtype` to its own line, asserted the DEFAULTS keys, and the scoring rehearsal used *exactly* the launch params.
  - Lesson: rehearse with the real hyperparameters, and override only the model names and paths.
- **Zero-shot bge.** Band AUC 0.56 / 0.64: it does not know what "same business" means on these near-duplicates. Blend gain ≤ +0.0005.
- **In-domain blend weights for an unseen country.**
  - India→US: bge −0.0016, e5 −0.0029.
  - The CE is not useless there (oracle weights +0.0014); the weights over-trust it. In-domain the fit puts ~0.63–0.71 on logit r, out-of-domain ~0.24–0.28.
- **Ensembles.** bge + s2P logit mean is +0.00041, below bge alone (+0.00059).
- **Qwen3-Reranker-0.6B.** Never ran: the job crashed before its phases, and the master then dropped it.
- **Local rehearsal in the real DLC, and local real-weight checks.**
  - The laptop downlink was 0.13–0.2 MB/s over WARP, and HF's xet transfer stalled at 0 bytes twice.
  - Replaced by three things:
    - the DLC build history read via ECR, which is why pyarrow is pinned;
    - an in-job self-check: batched vs single-pair logits, max Δ 0.094, passed on the real bge;
    - CPU/CUDA rehearsals with small models, including a forced-OOM micro-batch probe.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`probes/P13-aws-rerank/solution_p13.py`** is the SageMaker job. It is self-contained and shipped as solution.py next to sm_train.py.
  - `main(train_dir, test_dir, model_dir, out_dir, params)` runs the phases in `params['phases']`:
    - `bge_zs`, `bge_locoUS`, `bge_locoIndia`, `bge_pool`;
    - `qwen_*` (untested at scale);
    - `bge_ckpt`: scoring only, restores `params['ckpt']`.
  - Every r file goes to model_dir and S3 as soon as it is written.
  - Deadline-aware: a phase starts only if its measured-speed estimate fits, and training is capped in seconds.
  - Micro-batch OOM probe; OOM-halving scoring; `selfcheck` (batched = single-pair logits); `save` / `restore` (fp16, no frozen embeddings).
  - A10G: train 287–367 pairs/s at 11.8–15.5 GB; score 1,370–1,940 pairs/s at 4.4 GB; job RAM < 16 GB.
- **`probes/P13-aws-rerank/p13.py`** is the laptop side (docstring lists every command).
  - Data and launch: `build` (sm_data, 26 s, 1.5 GB), `tiny`, `launch` (sm_launch with our 3-file bundle + pinned reqs), `s3check`, `fetch`.
  - Local GPU scoring: `e5` / `e5fr` (e5-small controls on the GPU).
  - Evaluation, all on CPU: `pool <tag> [ctl] [d3|plain]` (the #5 yardstick, ~10 min, ~2 GB), `loco`, `xw`, `frproxy` (France procedure on LOCO, ~1.5 min).
  - Delivery: `frtest` (P6 threshold + red flags on the final files, 75 s, 1.1 GB), `testblend <tag> <c> [w]` (6–14 s per country, 0.7 GB).
- **`probes/P13-aws-rerank/requirements_sm.txt`**: transformers==5.17.0, tokenizers==0.23.2, pyarrow==25.0.1. The DLC already had pyarrow 25.0.1.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / size.** bge-reranker-v2-m3 is Apache-2.0 (HF card_data), 568M params. The only external input is its pretrained weights, downloaded from HF: no data, no API.
- **France is further out-of-domain than the proxy, and the blend changes France a lot.**
  - The CE and the GBDT disagree (one of r, p above 0.5, the other below) on **31%** of France band pairs. That compares with 10–12% on India/US test in-domain and 20–26% on the LOCO proxy targets.
  - France's band is 32% of its pruned pairs, vs 17–19% for India/US.
  - As a result the blend changes the answer of **15.7% of France S1** vs #6 (+21,604 / −23,168 pairs). Of that, 15.6% comes from the blend at a fixed t 0.78 and 3.5% from the threshold move alone.
  - The proxy's targets changed only 4.5% / 2.7% of S1.
  - So the +0.0015 to +0.0028 proxy gain is an extrapolation for France. Its direction is supported, its size is not.
  - The profile stays normal: 5.76% empty, 3.23 matches/S1.
- **France is validated on a proxy of two directions only.** France's shift is larger than India↔US: P14 found 35% missing state in France S2/S3, vs 3% in training. Limits on the downside:
  - the OOD weights put only 0.26 on logit r, vs 0.63 for India/US;
  - both directions and both CE sizes gain;
  - France keeps a label-free threshold, which this probe re-derived on the blended p.
- **Threshold change.** France's threshold moves from 0.78 to 0.74 because the blend tightens France's p (P6's rule).
  - In the proxy the same move happened (0.80 → 0.71 US→India) and it still gained +0.0028.
  - If the master prefers to change one thing at a time, the India/US-only fallback above is the clean variant: France = #6's.
- **GPU dependency.** The final pipeline needs bge-large inference on 1.66M pairs: ~18 min on an A10G, ~2+ h (estimated) on the laptop.
  - For the code package, the checkpoint plus `bge_ckpt` makes it reproducible without retraining. Retraining costs 76 min on an A10G, ~$2.2.
- **Holdout optimism** is the same as P4b's. The holdout shares the train S2/S3 corpus with the CE's training pairs; P4b argued the estimate is conservative.
- **Assumption.** The India/US weights were fit on both countries' band pairs together, like #5/#7, not per country.

## Open questions   (for the master; each with your best guess)
- **Does bge scoring go into the final code package, or do only its test outputs ship?**
  - The package must reproduce the output. That means the checkpoint (624 MB, or a retrain: 76 min on an A10G) plus a GPU for ~18 min of inference.
  - Best guess: ship `ce_bge_pool.pt` + the `bge_ckpt` scoring path. Document the A10G run and the laptop fallback (~2+ h, not measured).
- **France at 0.74 (P6's rule on the blended p) or 0.78 (#6's value)?**
  - Best guess: 0.74. It is the rule that tracked the oracle in P6's LOCO, and the proxy here applied the same threshold drop (0.80 → 0.71) and still gained +0.0028.
  - One upload of each would settle it on the public LB; France is ~15% of it.
- **Refit France weights once there are more labelled countries?** Only India/US exist, so w_ood is the mean of two directions. Best guess: keep the mean vector; the two directions agree on the r weight (0.24 / 0.28).
- **Worth the Qwen3 / bigger-model follow-up?** Best guess: no.
  - P11's oracle says the band's CE headroom is +0.0048; bge takes +0.0010 of it over #5.
  - The remaining gap is France (P14), not India/US capacity.
