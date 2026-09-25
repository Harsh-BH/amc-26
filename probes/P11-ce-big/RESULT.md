# P11-ce-big — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **No change reaches +0.0005.** The best is **2 epochs instead of 1** for the same e5-small on #5's exact training set: 0.98884 vs 0.98849,
  **+0.00035 ± 0.00007** (D3 yardstick). With the shipped rule (argmax, p ≥ t) it is **+0.00039 ± 0.00006** (6.5 SE).
- **Capacity barely helps.** e5-base (2.4× params) vs e5-small, same data and code, 1 epoch: +0.00008 ± 0.00006, not significant; epochs on the same code: +0.00028 ± 0.00006. The "matched" m2_big band loses −0.00016, and ensembles add ≤ +0.00002.
- **Recommendation:** swap `ce.pt` for the 2-epoch checkpoint. It is the same model with the same test cost, a 40 min retrain, and a +0.0004 gain at 6.5 SE. MERGE-2 adopted m2_big on exactly that evidence. Candidate #6 is staged and passes the validator. Do not buy e5-large for this: the capacity gain is small.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Adopt only if you apply MERGE-2's "certain, zero-complexity" ruling. It is below the brief's +0.0005 bar.**

What changes:
- `data/work/p11/ce.pt` (→ `ce_s2P.pt`, 43 MB, fp16, no word embeddings) replaces `data/work/p4b/ce.pt`.
- It is the same base (`intfloat/multilingual-e5-small`), the same format, and the same `rerank.load_model` / `score` path.

To reproduce the retrain in `src/rerank.py`:
- Call P4b's `train()` with `epochs=2`. Nothing else changes: bs 128, lr 6e-5, 500 warmup steps, linear decay, frozen word embeddings, max_len 64, length buckets.
- The data is also unchanged: the same 778,520 pairs, i.e. P4b's `trainband_*`, folds 4–9, 0.1<p<0.9 + 0.5× out-of-band.
- Time: 40 min on the RTX 3050. P4b's in-loop tokenisation would be ~50 min.

Blend and threshold, both refit on holdout half 0 of `prob_holdout_big`:
- Blend: **w = [0.617, 0.666, 0.261]**, band 0.02 < p < 0.98, India + US only.
- Shipped rule: **t = 0.71** for India / US (#5: 0.69). D3 would pick t 0.70 / t0 0.77.
- France is untouched: m2_big p, t 0.78 (P6's rule runs on m2_big p, which did not change).

Test files, ready for `submit_probs.py`:
- `data/work/p11/prob_test_{India,US}_blend.parquet`: [s1, sx, p = blended, p_gbdt = m2_big p]. Same rows and order as `p4/prob_test_*_big`; p equals p_gbdt outside the band.
- `data/work/p11/sub6/prob_{country}.parquet`: symlinks, with France = `p4/prob_test_France_big.parquet`.

Staged candidate **#6**, validator **PASS** (1,732,544 rows each):
```
.venv/bin/python code/business_entity_resolution/src/submit_probs.py --probs 'data/work/p11/sub6/prob_{country}.parquet' \
    --out output/p11_sub6 --t 0.71 --t-country France=0.78        # written: <worktree>/output/p11_sub6/
```

Test answer profile vs #5:

| | #6 (t .71) | #5 (t .69) | churn |
|---|---|---|---|
| India | 3.3772 matches/S1, 5.75% empty | 3.3777, 5.75% | 0.94% of S1 change (3,675 pairs added / 4,134 dropped) |
| US | 3.3793, 5.74% | 3.3794, 5.73% | 0.84% (2,770 / 2,880) |

France: 3.239 / 5.69%, identical to #5.

## Evidence         (table: ledger name | world | key metrics | verdict)
Yardstick, identical to #5's number:
- P4's `big._blend_eval(prob_holdout_big)`, 193,751 India+US band pairs. Blend w and D3 (t, t0) are fit on half 0 and reported on half 1.
- The paired SE is over half-1 S1, against #5's CE. The control reproduces **0.98849**.
- "p+r AUC" is the holdout band AUC of logit p + logit r.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| (control) | holdout-dense-half1 | #5's CE (P4b, e5-small, 1 ep): F0.5 **0.98849** (India 0.98935 / US 0.98791), t .69 / t0 .76. Band AUC r 0.9434, p+r 0.9625 (p alone 0.9220) | reference |
| trainband-m2big-f6-9+m2fast2-f3-5 | train | New band 0.02–0.98: m2_big (never saw folds 6–9) = 778,075 pairs, m2fast_fold2 on folds 3–5 = 605,086. m2_big only scored pairs m2fast was unsure of; 0.004% of the skipped pairs would have been in the band. 33 min CPU, 1.78 GB | keep (data) |
| ce-sA | holdout-dense-half1 | e5-small 1 ep on the NEW band, n 778,520 (= #5's count). **0.98833, −0.00016 ± 0.00006**. r alone better (AUC 0.9510 vs 0.9434), but p+r **worse** (0.9618 vs 0.9625): it learns what m2_big already knows. Val band AUC 0.958 (its own val) | drop |
| ce-sB-killed | train | e5-small, all 1.5M new-band pairs: killed at step 6,000/11,714 by the 20:41 suspend; not rerun, since sA showed the new band loses at equal size | inconclusive |
| ce-bP | holdout-dense-half1 | **e5-base** (278M), 1 ep, #5's exact data: **0.98864, +0.00015 ± 0.00006** vs #5's CE (541 up / 388 down); only +0.00008 ± 0.00006 vs s1P (same code). Band AUC r 0.9489, p+r 0.9640; P4b val AUC 0.9310 vs 0.9263. Train 49 min at 270 pairs/s, 4.07 GB VRAM (bs 64 × 2) | drop |
| ce-s2P | holdout-dense-half1 | **e5-small, 2 ep, #5's exact data: 0.98884 (India 0.98977 / US 0.98822), +0.00035 ± 0.00007** (559 up / 315 down). w [0.617, 0.666, 0.261], t .70 / t0 .77. Holdout band AUC r 0.9542, **p+r 0.9660** (#5: 0.9625). P4b val AUC **0.9364** (1 ep: 0.9263). 40 min, 2.41 GB VRAM | best (below bar) |
| ce-s2P_ep0 | holdout-dense-half1 | the same run's checkpoint after epoch 1 (LR still at 50%): +0.00014 ± 0.00007; val AUC 0.9275 | info |
| ce-s1P | holdout-dense-half1 | control for s2P: e5-small, 1 ep, #5's data, this probe's pipeline (= P4b's recipe, new seed stream): 0.98855, **+0.00007 ± 0.00005** vs #5's CE; val AUC 0.9282 (P4b 0.9263). Seed + pipeline noise is small | control |
| ce-like-for-like-epochs-capacity | holdout-dense-half1 | Same code, same data, paired: **2 ep vs 1 ep (e5-small) +0.00028 ± 0.00006** (4.7 SE); **e5-base vs e5-small (1 ep) +0.00008 ± 0.00006** (1.3 SE, not significant) | epochs yes, capacity ~no |
| ce-s2P-plain-rule | holdout-dense-half1 | shipped rule (argmax + p ≥ t, t tuned on half 0): ce5 t .69 = 0.98849 (= #5's shipped t); s2P t .71 = **0.98888, +0.00039 ± 0.00006** (552 up / 336 down) | keep |
| ce-ens_ce5_sA / _ce5_bP / _bP_sA / _ce5_bP_sA | holdout-dense-half1 | logit-mean ensembles: +0.00001 / +0.00020 / +0.00017 / +0.00021 (se 0.00005–6) | drop |
| ce-ens_s2P_bP / _ce5_s2P / _ce5_bP_s2P | holdout-dense-half1 | +0.00037 / +0.00028 / +0.00031. e5-base on top of 2-ep e5-small adds only +0.00002, at 2× test scoring | drop |
| ce-band-headroom-oracle | holdout-dense-half1 | Upper bound: r = truth gives 0.99332 (**+0.0048**); "halfway" r (p+r AUC 0.996) gives +0.0038. So the band is not saturated. Each CE step so far moves p+r AUC by ~0.0015 ≈ +0.00015 F0.5; **+0.0005 needs p+r AUC ≈ 0.967–0.968** | reference |

GPU used: ~2 h 55 min in total, including 20 min lost to the 20:41 suspend.

Costs, measured here:

| | train (RTX 3050 6 GB) | holdout band, 194k | **test band, 1.55M India+US** | VRAM |
|---|---|---|---|---|
| e5-small (P4b in-loop tokens) | 540 pairs/s | 1,344 pairs/s | ~20 min (P4b) | 2.4 GB |
| e5-small (this probe, pre-tokenised) | 620–990 pairs/s | 2,200–2,500 pairs/s | **India 6 min + US 3.5 min scoring**; ~15 min incl. writing the parquets | 2.4 GB |
| e5-base | 270–342 pairs/s (throttles at 86 °C) | 1,100–1,440 pairs/s | **~22 min GPU** (India 1,098 / US 1,340 pairs/s on 100k real test pairs); ~30 min end to end: **under the 1 h limit** | 4.1 GB (bs 64); 4.85 GB at bs 128 |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Band matched to the inference band (step 3).** m2_big's own 0.02–0.98 band on unseen folds, at #5's pair count, scores −0.00016 ± 0.00006.
  - The CE gets better on its own (band AUC 0.943 → 0.951) but becomes more correlated with the GBDT: p+r 0.9625 → 0.9618, and the blend weight on r drops from 0.715 to 0.647.
  - #5's odd-looking set is better because the weak model's wider band (+0.5× easy pairs) teaches text evidence that m2_big does not already have.
- **"More folds" (step 1 as briefed).** #5's CE was not trained on fold 3 only. P4b's `train()` used folds 4–9 (six folds) of a weak 300-tree fold-2 LightGBM (0.1<p<0.9, band AUC 0.81) plus 0.5× out-of-band pairs; fold 3 was validation. So more folds were already in.
  - The 1.5M-pair new-band run (sB) was killed by the 20:41 suspend and not rerun, because sA showed that band loses at equal size.
- **Capacity (step 2).** e5-base on identical data: +0.00015 ± 0.00006 vs #5's CE, but only **+0.00008 ± 0.00006 vs a 1-epoch e5-small from the same code** (s1P). p+r AUC rose by only +0.0015. The epochs lever (+0.00028 like-for-like) was worth more than 2.4× the parameters.
- **Ensembles.** They add ≤ +0.00002 over the best single CE, at 2× scoring.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
`probes/P11-ce-big/ce.py`. P4b's `texts`, `rows`, `blend`, `fit_blend` and `d3` are imported read-only. The docstring lists the commands.

Data and training:
- `band(country)` → `p11/trainband_<c>.parquet`: 15 / 18 min, 1.8 GB.
- `data()` → `tok_oob0.1_s0.npz` (deleted); `data_p4b()` → `p11/tok_p4b.npz` (#5's exact set, 35 s).
  - `tokenize(tok, ta, tb)` → flat int32 ids + offsets; `batch(ids, off, j, pad)` → padded tensors.
  - `check` asserts that they rebuild the tokenizer's own padded batch exactly, for both tokenizers.
  - e5-small and e5-base share one 250,002-token vocab (identical ids).
- `train(tag, base=small|base, epochs, src=p4b|all|big, n, bs, acc, lr)` → `ce_<tag>.pt` + `train_<tag>.json`. Peak 2.5 GB RSS.

Scoring and evaluation:
- `score(tag)` → `r_<tag>_holdout.parquet`.
- `eval_(tag)`: the #5 yardstick + paired se → ledger + `eval_<tag>.json`. 3 min, 1.3 GB.
- `ens(tag, parts)`: logit-mean ensembles.

Test side:
- `test(tag, country)` → `prob_test_<c>_blend_<tag>.parquet`, own process, under flock. Peak 2.98 GB, which is at the 3 GB cap: never run two at once.
- `test_r('ce5', …)` recovers #5's test r exactly by inverting #5's blend (asserted).

Other scripts:
- `plain.py <tag>`: the shipped-rule check.
- `auc.py <tag…>`: band AUCs.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences.** e5-small: MIT, 118M. e5-base: MIT (HF card `license:mit`), 278M; it stays in the HF cache (1.1 GB; delete it if not used). e5-large: MIT, 559.9M. No external data, and all scoring is offline (`HF_HUB_OFFLINE=1`).
- **France / unseen labels.** Nothing changed: the CE gate stays India + US (LOCO −0.0016), and France keeps m2_big p at t 0.78.
- **Training-seed variance** is not in the paired SE, which only covers holdout S1 sampling. The s1P control (a new 1-epoch run, same recipe) lands +0.00007 from #5's CE, so seed + pipeline noise is ~0.0001. The 2-epoch gain over it is +0.00028 ± 0.00006.
- **The ledger config of ce-bP says `oob: 0.1`.** That field is unused with src=p4b. The real set is #5's (0.5× out-of-band).
- **The t move 0.69 → 0.71** comes from the half-0 tuning of the shipped rule. The D3 tuning moves the same way (0.69/0.76 → 0.70/0.77).

## Open questions   (for the master; each with your best guess)
- **Is e5-large (560M) worth an A10G?** Best guess: **no.** Small → base bought +0.00008 ± 0.00006 like-for-like (not significant), so a further 2× parameters would likely add ≲ +0.0001. The measured lever is training length, not size.
  - VRAM with this recipe (fp32 master weights, AdamW, frozen word embeddings, bf16 autocast):
    - Fixed cost: 2.24 GB weights, 1.22 GB grads for the 304M trainable, 2.43 GB Adam states = **5.9 GB before any activation**. So it cannot fit the 6 GB GPU.
    - Activations, scaled from the measured e5-base (1.93 GB at micro-batch 64 × 2.67): ≈ 8.5 GB at mb 32, 11 GB at mb 64, **~16 GB at bs 128**. That fits a 24 GB A10G.
  - Time, scaled by non-embedding FLOPs (3.5× e5-base) and A10G ≈ 3.5× this RTX 3050:
    - Train: ~300 pairs/s, so ~45 min per epoch on 778k pairs; **2 epochs ≈ 1.5 h**.
    - Scoring: ~1,200 pairs/s, so test band ~22 min and holdout ~3 min. **≈ 2 h total** with setup.
    - Locally, even test scoring alone would take ~75 min (> 1 h budget).
- **A cheaper next step toward +0.0005?** Best guess: **e5-small × 3 epochs** on #5's data.
  - Cost: ~45 min on this GPU at 850–990 pairs/s; test scoring ~15 min.
  - The epoch trend was 1 → 2 ep = +0.00028. A third epoch might add +0.0001–0.0002 and reach the bar, or overfit. The half-split yardstick will tell.
  - e5-base × 2 epochs (~100 min + 30 min scoring) is the second choice, since capacity showed ~0.
  - Either way, stay with #5's training set, not the matched band.
- **Should the master adopt #6 now?** Best guess: yes, on the MERGE-2 ruling (+0.0004, 6.5 SE, zero complexity: only the checkpoint and t change).
