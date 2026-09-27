# P59-ce-bigdata — RESULT (final, 07:40)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **No.** A bge CE retrained on folds 1–9 (1.40× P13's pairs; fold 0 untouched) is a slightly better CE (holdout band AUC India 0.9710 → 0.9727, US 0.9564 → 0.9585). Under the shipped India/US combiner, though, it barely moves F0.5.
- **India/US gate fails.** Swapping it into the shipped combiner gives **+0.000024 ± 0.000024 LB-eq (1.0 SE)** cross-fit, against a bar of +0.00010 at 3 SE. Keeping both CEs gives +0.000045 ± 0.000025 (1.8 SE).
- **France fails.** In P38's LOCO gate the new LOCO CEs beat P13's by only +0.00024 / +0.00007 France-F0.5 (bar +0.0033). **Keep `ce_bge_pool.pt`. No sub59 was built. $9.0 spent.**

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing.** The India/US lines stay as they are in sub_clean (P50 fullzoo), and France keeps P13's CE.
- **Stop spending on the CE's training data.** The new CE is better in every p-region on #8c's argmax pairs:
  - AUC on the #8c band is +0.017 India / +0.014 US;
  - log-loss is lower in all regions.
  - The combiner still gains only +0.00002: it already extracts what a better CE adds.
  - A hard-example retrain (more out-of-band pairs) would therefore not plausibly reach +0.0001.
- **Artifacts kept, if anyone wants to reuse them:**
  - Checkpoint: `s3://sagemaker-114872709824-ap-south-1/amc26-p59/results/p59-pool-g6e2x-0927-0601/ckpt/ce_bge_pool.pt`. Same format as P13's, loads with `solution_p13.Job.restore`.
  - Holdout r_all (every A_<c> argmax pair, 878,543 rows): `data/work/p59/r_all_holdout_v1.parquet`.
  - LOCO holdout-band r: `data/work/p59/r/r_p59_loco{US,India}{,_mid}_holdout.parquet`.
  - Test r of US slice 1: `amc26-p59/results/p59-teus1-v1-0927-0654/r/`. The other test slices were stopped once the gate failed.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| **p59-iu-gate-zoo** | holdout, both halves (fit one, report the other) | Shipped combiner (P50 fullzoo: te44.FINAL + l_r_all + 16 zoo logits, 5 seeds, t 0.71), new − old l_r_all. **new: h0→h1 +0.000012 ± 0.000039, h1→h0 +0.000035 ± 0.000027, cross-fit +0.000024 ± 0.000024 (1.0 SE)**. India +0.000061 / +0.000025, US −0.000042 / +0.000061. newband (band r column swapped too): +0.000016 ± 0.000022. both (old + new columns): +0.000045 ± 0.000025 (1.8 SE). Old vs #34 h0→h1 +0.000681 (sub44r reference +0.000656) | **drop** |
| p59-iu-gate-nozoo | holdout, both halves | Same, on sub44r's combiner (no zoo columns): new +0.000058 ± 0.000032 / +0.000071 ± 0.000029, cross-fit **+0.000064 ± 0.000022 (3.0 SE)**, both halves > 0, but below the +0.00010 size bar; both +0.000065 ± 0.000023 | drop (robustness) |
| (region AUC, in RESULT only) | holdout argmax pairs by #8c p | India: p<.02 AUC 0.9967 → 0.9941 (log-loss 0.0037 → 0.0031), band 0.838 → **0.855**, p≥.98 0.977 → 0.983. US: 0.929 → 0.944, 0.790 → **0.804**, 0.979 → 0.984. Logit corr old/new r_all 0.9955 | – |
| p59-pool-band-auc | holdout band (78,079 India / 115,672 US, the same pairs) | AUC r India 0.9710 → **0.9727**, US 0.9564 → **0.9585**. Stack 0.9773 / 0.9682. Fold-1 val AUC 0.975 / 0.936. 2 full epochs, 16,998 steps, final loss 0.187 | keep (model) |
| p59-leakage-check | holdout vs the new train channel | 0 of 86,108 India / 129,215 US holdout S1 texts appear in CE training. SX-text overlap 3.5% / 4.6% (old CE 2.7% / 3.6%; the protocol every matcher shares) | clean |
| p59-locoUS-auc | holdout band | Trained on US folds 1–9 only: India AUC r **0.839** (P13 LOCO 0.821). India folds 1–9 only: US 0.828 (P13 0.829) | – |
| **p59-frgate-loco-new-vs-p13** | holdout-dense-half1 | P38's gate (P13's France procedure), new − old LOCO CE, mean weights: **US→India +0.00024 ± 0.00014, India→US +0.00007 ± 0.00009** France-F0.5 (xfit +0.00019 / +0.00011). Logit-mean ensemble +0.00013 / +0.00017. Bar +0.0033 in both directions | **drop (France)** |
| p59-launch | aws | 1,087,872 train pairs (India 407,586 / US 680,286) vs P13's 778,520. Pool on g6e.2xlarge (L40S) at 804 pairs/s; g5.4xlarge duplicate at 339 pairs/s, stopped | – |

**Cost: ≈ $9.0**, from BillableTimeInSeconds × the AWS Pricing API's APS3-Train rates:

| job | instance | seconds | cost |
|---|---|---|---|
| pool | g6e.2xlarge | 3,021 | $2.82 |
| pool duplicate | g5.4xlarge | 361 | $0.24 |
| locoUS | g6e.xlarge | 1,131 | $0.98 |
| locoIndia | g6.4xlarge | 2,368 | $1.31 |
| holdout scoring | g5.xlarge | 885 | $0.42 |
| test slices, stopped (4 jobs) | g5.4xlarge / g6.4xlarge / g6.2xlarge / g5.8xlarge | 1,269 / 1,267 / 1,268 / 610 | $2.70 |
| test slice (completed) | g5.2xlarge | 1,146 | $0.58 |

EBS and S3 costs are cents.

## What failed      (approaches tried and dropped, with the number that killed them)
- **More CE training data for India/US.**
  - 1.40× the pairs (folds 1–3 added) buys +0.0017 / +0.0021 band AUC.
  - Under the shipped combiner that is +0.000024 ± 0.000024 LB-eq. The bar is +0.00010 at 3 SE.
- **Swapping the band r column (a10's `r`) as well:** +0.000016. **Keeping both CEs:** +0.000045 (1.8 SE).
- **France transfer.** The more-data LOCO CEs gain +0.00024 / +0.00007 France-F0.5 over P13's. The gate needs +0.0033.
  - India→US is flat: AUC 0.828 vs 0.829. That is the direction that has predicted every France LB move so far.
- **g6e capacity.** P60 held every g6e type except on-demand 2xlarge and 16xlarge from about 06:15, and g6e.16xlarge never left Pending (17 min). Scoring moved to g5/g6.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`build59.py`**
  - `band <c>`: builds the folds 1–2 band. LightGBM with P4b's params, fit on a 30% sample of fold 3 → `data/work/p59/trainband12_<c>.parquet`. 30 s, ≤ 1.35 GB.
  - `chan`: adds P4b's trainband folds 3–9. val = 20k fold-1 band pairs; train = the rest, with P13's sampling (band + 0.5× out-of-band) → `sm_data/{train,val}_<c>.parquet`. Channel on S3: `amc26-p59/data/v1/`.
- **`chan59.py`**
  - `holdout`: texts for every A_<c> argmax pair → S3 `amc26-p59/data/ho/`. Texts are identical to P45's route (3,000-pair check).
  - `testkeys`: every test #8c argmax pair (6,596,109; the rows of r_all_test) → S3 `amc26-p59/data/cetest/`. That prefix also holds France's 1,332,223 pairs plus France texts. Local copies deleted; re-download to `data/work/p59/cetest/keys/` before `ship59.py rtest`.
- **`launch59.py <name> <instance> <channel> '<hp>' [solution]`**: sm_launch with P13's bundle. It skips the quota lookup, because ListServiceQuotas was being rate-limited.
- **SageMaker code runs unchanged:**
  - P13's `solution_p13.py`: phases `bge_pool`, `bge_loco{US,India}`, `bge_ckpt`.
  - P45's `solution_p45c.py`: test slices.
- **Scripts:**
  - `score59.sh <ckpt> <tag>`: scoring fan-out; instance types can be overridden through env vars.
  - `val59.sh <sub>`: validator `--check-ids` on ml.m5.xlarge.
  - `poll.sh`: job status.
- **`eval59.py <r_new> [arms] [zoo|nozoo]`**: the India/US gate (paired per-S1 F0.5, both directions, cross-fit), ≈ 5 min, ≤ 1.3 GB.
  - `eval59.py selftest` (old = new gives exactly 0) passes, and reproduces sub44r's +0.000656.
- **`ship59.py fit|rtest|probs|splice|profile|basecheck`**: the ship path onto sub_clean. It applies the diff vs the fullzoo decision, so P54's 14 dup copies are kept.
  - Self-check: refitting with the old r reproduces P50's fullzoo test decision exactly (0 adds / 0 drops, both countries).
  - `basecheck`: sub_clean's India/US = fullzoo + 14 dup pairs.
- **`fr59.py`**: P38's LOCO gate with the new LOCO CEs in the challenger slot. 80 s, 1.27 GB.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Assumption: "folds 1–5" read as every labelled fold except 0 (folds 1–9).**
  - Fold 0 is the #8c holdout, and it is excluded by construction.
  - trainband only covers folds 3–9, so folds 1–2 got a band model fit on fold 3. P4b's model is fit on fold 2, so it cannot label fold 2.
  - Its band is slightly narrower: 31k vs 33k band pairs per fold.
- **Data grew 1.40×, not 2.5×.** The brief asked for P13's recipe (band + 0.5× out-of-band), and I kept it. A 2.5× set (all out-of-band pairs) was not tried; see the region analysis for why it is unlikely to matter.
- **The holdout frame's competing train S1 (holdout = False rows) can now come from CE folds 1–3 too.** The scored S1 are clean, and the verdict is a drop, so this has no effect.
- Licence and size are unchanged: bge-reranker-v2-m3, Apache-2.0, 568M params.

## Open questions   (for the master; each with your best guess)
- **Is the both-CEs arm (+0.000045, 1.8 SE; +0.000065 at 2.8 SE on sub44r's combiner) worth a later upload?**
  - Best guess: no. Expected LB ≈ +0.00004, and it would need about 6.6M test pairs rescored (≈ 25 min on 4 GPUs) plus a combiner refit.
- **Should the new CE replace P13's in the France chain (P48 veto r < 0.2, P54 rescue r ≥ 0.99)?**
  - Best guess: no. The LOCO transfer gain is within noise, and India→US is flat.
