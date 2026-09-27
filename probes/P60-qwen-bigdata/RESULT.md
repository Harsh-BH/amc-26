# P60-qwen-bigdata — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **More labelled data does not improve Qwen as France's out-of-domain CE.** P38's LoRAs were warm-started on the 327,881 pairs they never saw (folds 1–3 of P59's channel), so each model has seen folds 1–9, 1.42× P38's pairs.
  - LOCO gate, new − old Qwen, France-F0.5 (mean-OOD mode): **US→India +0.000044 ± 0.000097, India→US +0.000057 ± 0.000074**. The bar was +0.0005 in both directions. **FAIL.**
  - Both results are under 1 SE. In India→US more S1 go down than up (224 vs 165).
- P59's from-scratch bge retrain on the same 1.40× data shows the same pattern (+0.00024 / +0.00007). Data quantity is not the France lever.
- **Keep P38's pooled Qwen (`p38-pool-0927-0145`).** No sub60 was built. The pooled job was stopped at step 1,100 of 2,561 once the gate failed. Total cost $7.6 of the $45 budget.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing from P60.**
  - France CE stays P38's pooled Qwen: `lora_qwen_pool.pt`, w = [0.8943, 0.3729, −0.0521].
  - The staged `sub38*` / `sub_clean` files are unaffected.
- Do not spend more GPU on a from-scratch Qwen retrain on folds 1–9. Two independent CEs (Qwen warm-start here, bge from scratch in P59) both give ≤ +0.0002 France-F0.5 from 1.4× data. The India→US out-of-domain band AUC does not move for either (Qwen 0.868 → 0.8675, bge 0.829 → 0.828).
- **Byproduct, unvalidated:** `output/q38c`.
  - What it is: sub_clean's exact chain (build38 clean on sub54 with hn .80 → P55 → dup) with **P38's** Qwen as the CE.
  - How it differs from P50's `sub38_clean_p55`: only the hn .80 extras (+3,472 pairs, 0 removed), since sub38_clean used hn .90.
  - Use it if the master wants the apples-to-apples Qwen version of sub_clean. It needs the validator (`--check-ids`) first. Otherwise delete it (98 MB).

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| **gate-qwen60_loco-vs-qwen38_loco** | holdout-dense-half1 | Setup: P38's gate (P13 France procedure, OOD blend weights, P6 empty-rate t), same holdout band pairs, P38's LOCO Qwen in the comparison slot. **new − old, mean: US→India +0.000044 ± 0.000097 (167 up / 178 down), India→US +0.000057 ± 0.000074 (165 / 224).** xfit: +0.00006 ± 0.00011 / +0.00013 ± 0.00009. Old+new logit-mean − old: +0.00008 / +0.00007. Over no-CE: old +0.00420 / +0.00257 (= P38's published numbers), new +0.00425 / +0.00263. Target band AUC: US→India 0.8575 → 0.8609, India→US 0.8681 → 0.8675. New mean-OOD w [0.8881, 0.3732, −0.0883] | **drop** |
| p60-loco-auc-vs-p38 | holdout band (78,079 India / 115,672 US) | Trained on India: India AUC r 0.969 → **0.9706**, US 0.869 → **0.868**. Trained on US: US 0.958 → **0.9596**, India 0.856 → **0.8608**. In-domain gains are +0.0016, out-of-domain −0.001 / +0.005. P59 fold-1 val: India-trained 0.972 / 0.838, US-trained 0.896 / 0.936 | drop |
| p60-chain-q38c | test-France | Setup: run60.sh on P38's pooled Qwen r. Chain: t 0.9868, recipe 834,521 → gib +1,453 → P48 −926 → P54 +3,319 → cls80 +1,183 → P55 −2,530 / +2,313 → dup +50. **vs sub38_clean_p55: 3,426 lines, +3,472 / −0** (the hn .8 extras only). 4 min, 1.33 GB | pass (chain works; `output/q38c`) |
| (early read, not logged) | holdout-dense-half1 | India→US with only the India model swapped: +0.00007 ± 0.00006 | – |
| (chan60) | train folds 1–3 | P59's train rows replayed (y and p equal). Folds 1–3: India 120,675 (true .366), US 207,206 (true .321). Folds ≥ 4 in P59's channel: 286,911 / 473,080 (P13's: 294,656 / 483,712) | – |
| (gate selftest) | holdout-dense-half1 | old Qwen vs itself: gain exactly 0 in every mode, 86 s, 1.26 GB | pass |

**AWS** (ap-south-1, `--profile harsh`; billable s × on-demand rates from P38: g6e.xlarge $3.13/h, g6e.2xlarge $3.37/h). **Total ≈ $7.6.**

| job | instance | role | billable s | $ |
|---|---|---|---|---|
| p60-locoUS-od1-0927-0630 | g6e.xlarge on-demand | LOCO US: 207,104 pairs at 85.7 pairs/s (2,331 s), then the holdout band and val | 4,114 | 3.58 |
| p60-locoIndia-sp-0927-0622 | g6e.2xlarge **spot** | LOCO India: 120,576 pairs at 66.5 pairs/s (1,857 s). 3,593 s of training time, 40% spot saving | 2,156 | 2.02 |
| p60-pool-od2-0927-0654 | g6e.2xlarge on-demand | pooled warm start, 77.7 pairs/s. **Stopped** at step ~1,100 of 2,561 after the gate failed; the partial LoRA is on S3 | 2,118 | 1.98 |
| 9 other copies (g6e.4x / 8x / 16xlarge on-demand and spot, g6e.xlarge spot) | – | capacity race, stopped while pending | 0 | 0 |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Warm-starting Qwen on 1.42× labelled pairs:** +0.00004 / +0.00006 France-F0.5 against a +0.0005 bar, and both are under 1 SE.
- **Old + new Qwen logit-mean ensemble:** +0.00008 / +0.00007. Two LoRAs of the same base are too similar to ensemble.
- **Getting g6e capacity.** Every g6e size except the two P59 had just released waited with "insufficient capacity" / "waiting for capacity", for 10–40 min, on-demand and spot alike. What worked:
  - Spot g6e quota is separate from on-demand (1 per size), so each role got several copies and the first one to start was kept.
  - Resubmitting on a type the moment another probe's job finished on it got an instance within 3–9 min.
- **Not tried: a from-scratch run on all 1.09M pairs.** It needs about 4 h of training on one L40S (76 pairs/s), which does not fit MaxRuntime ≤ 4 h plus France scoring. P59's from-scratch bge result is the stand-in, and it also fails.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`chan60.py`**
  - What it does: replays P59's `build59.chan` selection on (fold, inband, is_true, p) and takes the rows of P59's `train_<c>.parquet` with fold ≤ 3.
  - Checks that y and p match P59's file row for row.
  - Output: `data/work/p60/sm_data/train_<c>.parquet`. 2 s, < 0.3 GB.
- **`launch60.py <name> <instance> <channel> '<hp>' [max_run ≤ 14400]`**
  - Runs P38's `solution_p38.py` (unchanged) with P13's bundle. `SPOT=1` submits on spot.
  - Warm start = P38's `ckpt` hyperparameter plus a non-empty `train` list: `restore()` runs before `train()`.
  - Appends every job to `data/work/p60/jobs.txt`.
- **`p60.py`**
  - `fetch <job>`: r files into `data/work/p60/r`, with P38's and P13's LOCO files symlinked in.
  - `gate`: `p38.gate` with P38's LOCO Qwen as the comparison. Writes this probe's own ledger row. 60 s, 1.23 GB.
- **`run60.sh <job> <a,b,c>`** (optional `RQ=<r file> S=<name>`)
  - Steps: build38 blend → tpick → post → clean (sub54, hn .8, v .2, u .99) → P55 (`v=0.6,scope=band,u=0.95`) → build54 dup.
  - Outputs `output/<S>_fr` and `output/<S>` (sub54's candidate_pairs hard-linked). 4 min, 1.33 GB. Validator not included.
- `poll.sh` (job status + events) and `waitstart.sh "<jobs>" [s]` (returns when the first job leaves Pending/Starting).
- **S3:**
  - Channels: `amc26-p60/data/{loco,pool}/`. loco = new train + P59 val + P13 holdout. pool = new train + P59 val + P38's France band (319,752) + out-of-band audit pairs (29,376).
  - LoRAs: `amc26-p60/results/<job>/ckpt/`. `lora_qwen60_loco{US,India}.pt` are final; `lora_qwen60_pool_partial.pt` is from step ~1,100 of 2,561.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence and data:** unchanged from P38. Qwen3-Reranker-4B is Apache-2.0 with 4.02B params. No external data.
- **Assumptions:**
  - "Folds 1–5" means every labelled fold except 0 (P59's reading; its channel is folds 1–9). The data grows 1.40–1.42×, not 2.5×, because P13's recipe keeps the band plus 0.5× as many out-of-band pairs.
  - The warm start (lr 5e-5, one pass over the new pairs, fresh warmup + linear decay) stands in for a from-scratch run. The same recipe was used for LOCO and for the pooled model, so the gate measured what would have shipped. A from-scratch run might differ, but P59's bge result says not by +0.0005.
- **Where I worked:** the main checkout, not a worktree (P38 and P59 did the same). Everything is new, under `probes/P60-qwen-bigdata/`, `data/work/p60/` and `output/q38c`. No existing file was modified.

## Open questions   (for the master; each with your best guess)
- **Is any more CE data/training worth it for France?**
  - Best guess: no. The out-of-domain band AUC stays at ~0.86 for Qwen and ~0.83 for bge whatever the data amount. The limit is domain shift, not data.
  - France gains are more likely to come from rule and combiner work (P55 / P61) than from CE retraining.
- **Should `output/q38c` replace `sub38_clean_p55` as the Qwen fallback?**
  - Best guess: only if sub_clean's hn .80 beats hn .90 on the LB. q38c is exactly sub_clean's chain with Qwen as the CE.
  - It needs the validator (`--check-ids`) before any upload.
