# P38-qwen4b-fr — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Qwen3-Reranker-4B + LoRA beats bge-reranker-v2-m3 as France's out-of-domain CE, in both LOCO directions at 8 SE, but misses the brief's bar.**
  - France procedure, mean-OOD weights. Qwen − bge: **+0.00141 ± 0.00017** (US→India) and **+0.00109 ± 0.00014** (India→US) France-F0.5, ≈ +0.0002 LB-eq. The bar was +0.0005 LB-eq.
  - Unseen-country band AUC: 0.857 / 0.868 (bge 0.821 / 0.829).
  - India/US: no gain. Qwen stacked on #34 gives +0.00007 ± 0.00004 (1.9 SE), below the +0.00015 at 3 SE bar.
- **Three files staged, validator `--check-ids` PASS on all:**
  - `output/sub38`: #37_gc70 with only the CE swapped. Expected +0.0002 LB.
  - `output/sub38_full`: the full France stack with Qwen, on sub50e.
  - `output/sub38_clean`: the master's clean rule subset with Qwen, on sub54. P50 adds P55 + dup.
- **Qwen audit of the France rules:** P46 tier, P40 xf and P41 gib are what both CEs reject. They are CE-blind by construction. P46 is the prime suspect for sub54's LB shortfall.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**France CE → Qwen3-Reranker-4B + LoRA.**
- Model: pooled India+US, 1 epoch, job `p38-pool-0927-0145`. LoRA at `s3://sagemaker-114872709824-ap-south-1/amc26-p38/results/p38-pool-0927-0145/ckpt/lora_qwen_pool.pt`.
- Band and blend: the same band as bge (0.02 < p_gbdt < 0.98). Blend σ(a·logit p + b·logit r + c) with Qwen's own mean-OOD weights **w = [0.8943, 0.3729, −0.0521]** (bge's: [0.9245, 0.2595, 0.0515]).
- Band r: `data/work/p38/r/r_qwen_pool_test_France.parquet`, 319,752 pairs (also on S3 under `…/p38-pool-0927-0145/r/`).
- Everything downstream is unchanged:
  - t by #9ab's 6.71% pre-rule empty rate. It stays **0.9868**: the empty rate at that t is set by pairs above the band.
  - The P30 and P48 vetoes keep bge r.

**Files** (India/US byte-identical to their base; validator `--check-ids` PASS on SageMaker):

| folder | base | France | validator job, md5 | expected LB |
|---|---|---|---|---|
| **`output/sub38`** | #37_gc70 (LB 0.986304) | only the CE: rescue 1,043 (bge 1,205), gc +9,155 (10,084). 814,127 pairs, 6.085% empty, 3.138/S1. 6,885 France S1 differ | p38-val-0927-0536, 29d85ea8… | ≈ 0.98647–0.98651 (+0.00016–0.00021) |
| `output/sub38_full` | sub50e (France = sub48) | recipe 834,521 → gib +5,759 → xf +4,632 → P46 +2,976 → P48 veto −1,256 → P47 cls80 +1,694. 3.253/S1, 6.05% empty. 13,667 France lines differ | p38-val-0927-0536, 2f416b24… | the same stack with bge + ≈ 0.0002 |
| `output/sub38_clean` | sub54 | P42 (hn ≥ 0.9) 828,923 → gib +2,586 → P48 veto on **Qwen r < 0.2** −926 → P54 rescue on **Qwen r ≥ 0.99** +3,914 → cls80 +1,183. No P46, no P40, no dup copies. 3.221/S1, 6.02% empty. 24,720 France lines differ | p38-val54-0927-0549, b4fe0d20… | P50 runs P55 + dup on it |

**Rebuild:**
- `sub38`: `probes/P38-qwen4b-fr/run38.sh p38-pool-0927-0145`. It runs fetch → `build38.py blend` → `tpick` → `post` → `sub`, then the validator.
- `sub38_full`: `build38.py full data/work/p38/post_qwen.parquet data/work/p38/blend_qwen.parquet 0.9868 output/sub38_full sub50e q`
- `sub38_clean`: `build38.py clean data/work/p38/post_qwen.parquet data/work/p38/blend_qwen.parquet data/work/p38/r/r_qwen_pool_test_France.parquet 0.9868 output/sub38_clean sub54 c`

**run.py:**
- Add `RECIPES["sub38"] = dict(RECIPES[<base>], w_france=[0.8943, 0.3729, -0.0521])`.
- Put `p38/r/r_qwen_pool_test_France.parquet` first in France's test r seed (first file wins).
- Re-scoring needs a ≥ 24 GB GPU: the 4B weights are 8 GB in bf16, so the laptop's 6 GB RTX 3050 cannot run it.
  - Use `solution_p38.py` in scoring-only mode: `train: []`, `ckpt` = the LoRA above.
  - It takes ~34 min for the band on one L40S (158 pairs/s).

**India/US:** keep #34.

**France rules, Qwen's verdict** (details under Evidence):
- Drop or test-remove **P46 tier** first (3,185 pairs; only 1.9% get Qwen r ≥ 0.5), then **P40 xf** (30%).
- Keep P55, P47, P54, P48 and the t .99 removals.
- The hn bar 0.8 → 0.9 is marginal: hn extras at p .8–.9 get 61% r ≥ 0.5, the LB-proven gc70 adds 62%.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| **gate-qwen_locoUS_holdout+qwen_locoIndia_holdout-vs-bge_locoUS_holdout+bge_locoIndia_holdout** | holdout-dense-half1 | Setup: P13's France procedure (source in-domain blend, OOD weights on the target, shipped rule, P6 label-free t), per-S1 paired on the target's half 1, same pairs and 1 epoch as bge. **Qwen − bge, mean-OOD w: US→India +0.00141 ± 0.00017 (744 up / 385 down), India→US +0.00109 ± 0.00014 (969 / 411).** Cross-fit w: +0.00130 / +0.00106. Over no-CE: bge +0.0028 / +0.00148, Qwen +0.0042 / +0.00257, bge+Qwen logit-mean +0.00374 / +0.00239. Target band AUC r: bge 0.821 / 0.829, **Qwen 0.857 / 0.868**. Qwen w [0.8943, 0.3729, −0.0521] | drop vs bar (+0.0033); consistent → pooled + staged |
| (LOCO job AUCs, log.jsonl) | holdout band | Trained on India: India AUC r 0.969, US 0.869 (stack 0.943). Trained on US: US 0.958, India 0.856 (stack 0.956). 1 epoch each: US 483,712 pairs at 90 pairs/s (5,502 s), India 294,656 at 66 pairs/s (4,526 s) | – |
| (pooled job AUCs) | holdout band, val | In-domain AUC r: India 0.971 (bge pooled 0.971), US 0.960 (0.956); val India 0.965. 778,496 pairs, 1 epoch, 76.6 pairs/s, 9,946 s, loss 0.182. France band scored at 158 pairs/s | – |
| **iugate-qwen_pool-on-34** | holdout-dense-half1 | Setup: Qwen r stacked on #34's out-of-fold p′ (P36 base), band argmax rows, refit + t on half 0. India +0.000059 ± 0.000062, US +0.000123 ± 0.000064. **LB-weighted +0.000074 ± 0.000038 (1.9 SE).** bge control −0.000027 ± 0.000028 | **drop** (bar +0.00015 at 3 SE) |
| p38-sub38 / p38-sub38_full / p38-sub38_clean | test | Stage counts as in Recommendation. All three validator `--check-ids` PASS; 0 non-France lines changed vs base | keep (staged) |
| p38-chain-repro-gc70 | test-France | `build38.py repro`: bge inputs through the chain = **#37_gc70 byte for byte** (t 0.9868, rescue 1,205, gc 10,084) | pass |
| p38-chain-repro-sub43 | test-France | on43 on bge = #43_r90 to **2 rows**. Density rule gives t 0.9900; gib 5,742 = sub41; xf 4,692 = sub40x_t70 | pass |
| (reprofull) | test-France | full() on bge vs sub48 + P47 cls80: **23 France rows differ** out of 259,452. Cause: P48 veto 1,159 vs 1,138, since the r lookup is P31's bge union, not france.npz | pass |
| p38-iu-repro-34 | test | #34's India/US lines rebuilt from P34's prob files = #37_gc70's: 0 of 1,453,186 rows differ | pass |
| **p38-qwen-audit-sub54-vs-gc70** | test-France | Changed pairs by rule; Qwen r ≥ 0.5 share (in + out of band) / est precision on the band calibration (see the table below) | inconclusive (CE-blind rules) |
| (gate code check) | holdout | The bge arm reproduces P13's +0.0028 / +0.00148 exactly | pass |
| (rehearsal) | sm_tiny, CPU | Exact bundle end to end with a tiny random Qwen3. S3 restore reproduces r (max \|Δr\| 0) | pass |

**Qwen audit of sub54's France changes vs #37_gc70** (`diag54.py`; bge chain rebuilt in stage order).
- Calibration: P(true | Qwen r) on the LOCO unseen-country band, US→India / India→US: r < .05: .07 / .04; .9–.97: .82 / .52; ≥ .97: .96 / .88.
- The calibration understates absolute precision, so compare with the LB-proven adds rather than 0.68–0.75.
- Pairs are shown as n, share with r ≥ 0.5 over all pairs, share with r ≥ 0.9, and est precision (band pairs).

| rule | n | r ≥ .5 | r ≥ .9 | est prec |
|---|---|---|---|---|
| gc70 base accepted (baseline, 50k scored) | 802,584 | 87% | 78% | – |
| P55 adds (sub55 vs sub48) | 4,830 | 82% | 69% | 0.683 |
| P47 cls80 | 1,617 | 73% | 57% | 0.565 |
| P42 hn extras, p .98–.987 / .95–.98 | 7,063 / 5,874 | 77% / 72% | 63% / 46% | 0.568 (≥ .95) |
| (LB-proven) rescue90 adds | 1,205 | 68% | 49% | 0.506 |
| P42 hn extras, p .90–.95 | 5,159 | 67% | 36% | 0.462 |
| P54 rescue u .99 | 7,418 | 64% | 50% | 0.522 |
| (LB-proven) gc70 gc adds | 10,084 | 62% | 38% | 0.433 |
| P42 hn extras, p .80–.90 | 5,184 | 61% | 28% | 0.406 |
| P41 gib | 1,434 | 66% | **0.7%** | 0.288 |
| P40 xf_noaddr | 4,672 | **30%** | 0% | 0.194 |
| P46 noise tier | 3,185 | **1.9%** | 0.2% | 0.076 |
| removed: t .99 | 1,379 | 55% | 30% | – |
| removed: P55 vetoes | 3,198 | 49% | 17% | 0.314 |
| removed: P48 veto | 1,075 | 23% | 12% | 0.204 |

- hn extras by gc70 group size (1 / 2 / ≥ 3 members): est 0.50 / 0.50 / 0.49. The group state doesn't matter.

**AWS jobs** (ap-south-1, `--profile harsh`, on-demand; Pricing API rates): **total $25.41**.

| job | instance | purpose | billable s | $ |
|---|---|---|---|---|
| p38-smk-g6e2-0926-2338 | ml.g6e.2xlarge | smoke: real 4B, tiny data, every phase; filled the S3 weight cache | 359 | 0.34 |
| p38-smk-g52-0926-2340 | ml.g5.2xlarge | smoke. A10G trains at 21.5 pairs/s, too slow | 514 | 0.26 |
| p38-locoUS-0926-2346 | ml.g6e.2xlarge | LOCO: train US → holdout band | 7,279 | 6.81 |
| p38-locoIndia-0926-2346 | ml.g6e.xlarge | LOCO: train India → holdout band | 6,292 | 5.47 |
| p38-pool-0927-0145 | ml.g6e.xlarge | pooled → France band, India/US holdout, val | 13,863 | 12.05 |
| p38-diag-0927-0544 / diag5 | g6e.2xlarge / g5.2xlarge (stopped) | 29,376 out-of-band audit pairs | 384 / 195 | 0.46 |
| p38-val-0927-0200 / -0536 / val54-0549 | ml.m5.xlarge | official validator `--check-ids` | 154 / 207 / 154 | 0.03 |
| p38-smk-g6e / smk-g5 (12xlarge), smk-g6e4, pool2 | – | stopped while waiting for capacity | 0 | 0 |

## What failed      (approaches tried and dropped, with the number that killed them)
- **The brief's gate.** Qwen − bge is +0.0014 / +0.0011 France-F0.5, below +0.0033 (+0.0005 LB-eq) and far below +1 pt.
- **bge + Qwen logit-mean ensemble.** +0.0009 / +0.0009 over bge, below Qwen alone.
- **India/US arm.** +0.000074 ± 0.000038 (1.9 SE) on top of #34.
- **Multi-GPU.** ml.g6e.12xlarge, ml.g5.12xlarge and ml.g6e.4xlarge never got capacity (waited 12+ min each). The DDP path in solution_p38.py is untested on real multi-GPU. Every run used one L40S.
- **A10G.** 21.5 pairs/s training for India, 4× slower than an L40S. Not used.
- **Pseudo-label pooled variant.** Skipped on the master's call. P27's 27 GBDT pseudo-label arms each split the two LOCO directions.
- **Density-matching P54's u for Qwen.** Qwen needs u ≈ 0.87 to rescue as many pairs as bge at 0.99, and Qwen r in [.9, .97) is only .52–.82 true on LOCO. u stays 0.99 (3,914 pairs).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`solution_p38.py`** is the SageMaker job, shipped as solution.py next to sm_train.py. One model per job.
  - Flow: `train` countries → `score` files → `r_<tag>_<file>.parquet` (s1, sx, r), plus log.jsonl and `ckpt/lora_<tag>.pt` (LoRA + head, 132 MB; partial every 20 min).
  - Scoring-only: `ckpt` set and `train: []`.
  - Model: Qwen3-Reranker-4B body, bf16, frozen. LoRA r16/α32 on q,k,v,o,gate,up,down (33M params). Head initialised to W[yes]−W[no], plus a bias.
  - Prompt without the system turn, maxlen 160.
  - Weights come from the S3 cache `amc26-p38/weights/Qwen3-Reranker-4B/` (~20 s), falling back to HF.
  - Safety: early smoke-write, selfcheck, deadline-capped training, and a `score_speed` planning override.
  - DDP (mp.spawn + NCCL) when there is more than one GPU. Untested.
  - Speed on one L40S: train 66–90 pairs/s at 28–39 GB; score 136–177 pairs/s.
- **`p38.py`**:
  - `fetch <job>`
  - `gate <qwen_US> <qwen_India>`: France-procedure LOCO, bge vs Qwen vs ensemble. 40 s, 1.3 GB.
  - `iugate [tag]`: the India/US arm on #34. 5 s. Writes `iuspec_<tag>.json` for `build38.py iusub`.
- **`build38.py`** (1.3–1.45 GB):
  - `blend`, `ensr`, `tpick`, `post` (P31's), `sub`, `on43`, `full`, `clean`, `iusub`.
  - Repro checks: `repro`, `repro43`, `reprofull`.
  - Composes P31, P37, P41, P40, P42, P46, P47 and the P48 / P54 rules as functions on a France frame. `p37.fr_load` is monkeypatched so P41 / P40 read the right lists.
- **`diag54.py pairs|report`**: the rule audit. Outputs `data/work/p38/diag54.json` and `diag54_pairs.parquet`.
- **Helpers:**
  - `solution_val.py`: official validator `--check-ids` on ml.m5.xlarge, channels `amc26-p38/val/` (gc70 candidates) and `val54/` (sub54 candidates).
  - `run38.sh`: end-to-end sub38.
  - `rh.sh`: local CPU rehearsal.
  - `poll.sh`: job events.
- **Unused data:** India/US test argmax-band texts (959,658 pairs in 3 shards) are staged at `s3://…/amc26-p38/data/iu_test/`. Not scored, since the India/US gate failed.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / size.** Qwen3-Reranker-4B is Apache-2.0 (HF card_data) with 4,021,784,576 params. Pretrained weights are the only external input. No external data, no API.
- **France is a proxy extrapolation.** The +0.0002 LB-eq comes from two LOCO directions on the m2_big base, not from France's hn-matcher chain.
  - On the real chain the CE interacts with tier / bxneg / veto / P54. Qwen's band blend moves 6,885 France S1 vs gc70.
  - The profile holds: 6.085% empty, 3.138/S1.
- **Assumptions:**
  - The veto keeps bge r in sub38 / sub38_full, since P48's τ 0.2 was validated on bge.
  - sub38_clean uses Qwen r for P48 (< 0.2) and P54 (≥ 0.99) at bge's thresholds, as the master specified. Neither threshold was re-validated for Qwen.
  - P47's hard-coded base-stage t 0.99 is overridden to on43's density t, which comes out as 0.99 anyway.
- **Audit caveats:**
  - The calibration comes from LOCO band pairs (single-country models), so absolute precision is understated. Only relative ranking against the LB-proven adds is meaningful.
  - Rules whose pairs differ from the S1 by construction (P46 noise word, P41 brand, P40 empty address) are CE-blind. A low r there is not proof of falseness.
- **Reproducibility.** Re-scoring France needs a ≥ 24 GB GPU (~34 min on an L40S). The laptop cannot run it: 6 GB GPU, 8 GB weights.
- **Credits.** $25.41 spent. MaxRuntime was capped at 4–5 h because the account closes at $0.

## Open questions   (for the master; each with your best guess)
- **Upload sub38 (the clean CE swap) or fold Qwen into the final stack?**
  - Best guess: fold it in if a slot allows only one test. sub38_clean or sub38_full with P55 carries the same ~+0.0002.
  - sub38 alone is the clean read of the CE, at +0.0002 over gc70.
- **Is P46's tier really true on France?** Both CEs reject 98% of its pairs, but the CEs never saw French noise words.
  - Best guess: test sub54 − P46 in one slot. If the LB rises, the lens transfer was wrong.
- **Should P48 / P54 thresholds be re-fit for Qwen r?**
  - Best guess: yes, via P48's / P54's LOCO frames with the qwen_loco r files, which are in `data/work/p38/r/`.
  - Until then, Qwen's 0.99 is the stricter, safer rescue.
- **Would a 2nd epoch or a larger LoRA help?** The pooled model saw 1 epoch; bge had 2.
  - Best guess: small, and each run costs ~3 h on one L40S. Not worth it this late.
