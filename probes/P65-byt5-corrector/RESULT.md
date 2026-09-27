# P65-byt5-corrector — RESULT (final, 27 Sep 14:50)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **France: the gate fails. Do not build sub65.**
  - Setup: ByT5-small correctors trained on India/US true pairs, with their corrected-SX-vs-S1 sims added as band-only columns to P55's combiner.
  - They lower the France-chain analogue in the France-like direction: **India→US −0.00204 ± 0.00014** (conly set, shipped cell, both halves < 0), against US→India **+0.00088 ± 0.00014**. The gate needed India→US ≥ +0.0003.
- **Why:** the corrector is strongly **in-domain only**.
  - On its own country's band pairs, name-ratio AUC rises by +0.15 (India) / +0.16 (US), and US token-set AUC by +0.12.
  - On the other country, the gain is ~0 or negative. It learns the training country's canonical vocabulary, not a country-free "undo the noise" map. France would be out of domain for any India/US-trained corrector.
- **India/US (arm c, in-domain correctors): the gate fails too.** sub54's combiner + corrected sims on P57's full holdout frame, half-split: **LB-eq −0.0000007 ± 0.000023** (h0→h1 −0.000022, h1→h0 +0.000021). The in-domain AUC gain is already priced in by M2 / CE / the zoo.
- **Adopt nothing.** No p65 SageMaker job is running (≈ 5 GPU-h in total, ≈ $9 of the $35 budget).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing, for France or India/US.** No sub65 and no SUBMISSIONS row. `output/sub66_final` (LB 0.987391) stays the final candidate.
- Do not add raw SX-vs-S1 string sims to P55's combiner either: −0.00108 / −0.00117 at the shipped cell.
- If anyone retries a learned corrector for France, it must be **in-domain for France**: a denoiser trained on France S1 records with synthetic noise (P40's generator; synthetic pairs from provided records are allowed). An India/US-trained one cannot carry the signal (see Evidence, band AUC).

## Evidence         (table: ledger name | world | key metrics | verdict)
dF = France-analogue F0.5 of P55's combiner + corrector columns minus P55's own 20-feature combiner, both fit on the source's LOCO frame (bag of 3 seeds) and scored on the target. The shipped cell is veto band c < .6 + add free c > .95; veto06 is the veto alone. The base reproduces P55: +0.005268 (India→US) and +0.004640 (US→India) over the recipe.

| ledger name | world | key metrics (India→US / US→India) | verdict |
|---|---|---|---|
| p65-loco-combiner-strict | holdout-loco | **conly (pre-declared): −0.002044 ± 0.000136 / +0.000884 ± 0.000137** (h0 −0.00209 / +0.00082, h1 −0.00200 / +0.00095). delta: −0.001831 / +0.001464. corr (raw+corr): −0.00240 / +0.00051. raw: −0.00108 / −0.00117. veto06 conly: −0.001266 / +0.000614. Rule arm (no learning; bars picked on source): −0.00050 / −0.00003 | **drop** |
| p65-iu-sub54-plus-corr | holdout-full-halfsplit (P57 A_full) | sub54 (P57's cached boosters) vs sub54 + the 7 conly columns, refit on the same half with 5 seeds. Columns on IU band rows only (0.02 < #8c p < 0.98: 10,063 India / 23,660 US rows, 100% decoded by the in-domain corrector). LB-eq h0→h1 −0.0000221 ± 0.0000346 (India −0.000073, US +0.000032); h1→h0 +0.0000208 ± 0.0000291 (India +0.000022, US +0.000027); **mean −0.0000007 ± 0.0000226**. Lead's gate is ≥ +0.00015 at 3 SE, both halves > 0 | **drop** |
| p65-loco-combiner-oodfit | holdout-loco | Fit frame's columns come from the OTHER country's corrector (the France-build situation; not strict LOCO). conly −0.000776 ± 0.000109 / +0.000373 ± 0.000125. delta −0.000124 / +0.000938 | **drop** |

**Band-row AUC of single columns, true vs false** (`band_auc.json`; LOCO frames, 0.02 < g < 0.98):

| pairs / corrector | name ratio raw → corrected | addr ratio raw → corrected | token-set raw → corrected |
|---|---|---|---|
| India / India-trained (in-domain) | 0.626 → **0.780** | 0.414 → 0.459 | 0.547 → 0.594 |
| India / US-trained (out of domain) | 0.626 → 0.618 | 0.414 → 0.412 | 0.547 → 0.590 |
| US / US-trained (in-domain) | 0.384 → **0.545** | 0.565 → 0.645 | 0.537 → **0.656** |
| US / India-trained (out of domain) | 0.384 → 0.439 | 0.565 → 0.544 | 0.537 → 0.532 |

- Out of domain, the corrector also moves text away from S1: the mean address ratio falls 0.65 → 0.50 (India pairs, US model) and 0.70 → 0.62 (US pairs, India model).
  - A combiner fit where the corrector works learns to expect high corrected sims on true pairs, then vetoes true pairs where it does not.
  - India→US veto false-share: 0.53 with the conly columns vs 0.74 for the base.

**Protocol:**
- **Frames:** P42/P55's LOCO frames (target fold-0 SX-argmax pairs, source-only M2 g + CE r), P55's features and bag, and P55's reference = recipe + P48 veto.
- **Correctors:** trained on source-country true pairs with S1 not in fold 0 (400k sampled). Input `name | addr` (raw SX) → target `name | addr` (its S1), 192 / 160 bytes max, AdamW 5e-4 (US) / 3.5e-4 (India), bf16 autocast, capped at 30 min.
  - US: 6,126 steps × 64, final loss 0.147. India: 3,014 steps × 32, loss 0.219.
  - Greedy decode.
- **Columns** on band rows only, pre-declared before any result:
  - nm_c, ad_c (rapidfuzz ratio of corrected name / address vs S1) and ts_c (token_set_ratio, full text);
  - cd = 1 − ratio(raw SX, corrected);
  - ex_c = corrected name == S1 name;
  - dnm, dad = corrected minus raw ratio.
  - Text normalisation is casefold plus Unicode punctuation/symbol classes → space, so Indic and accented letters survive.
- **Directions:** A→B uses the A-trained corrector on both frames.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Corrector columns in P55's combiner (strict LOCO):** India→US −0.00204 (conly), −0.00183 (delta only), −0.00240 (raw + corr).
- **Same, with out-of-domain corrector columns on the fit side too:** India→US −0.00078 (conly), −0.00012 (delta).
- **Raw string sims alone:** −0.00108 / −0.00117. Their scale is country-specific.
- **Rule arm** (veto band pairs whose corrected name AND address both disagree; add free band pairs with an exact corrected name): ≤ 0 in both directions for every (v, u) cell.
- **Engineering notes:**
  - byt5-small ships only `pytorch_model.bin`; the first launch filtered it out.
  - bs 64 OOMs on a 24 GB A10G; use bs 32.
  - Decoding runs at 270–450 rows/s per GPU. Decode only the rows you use (band first).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `prep65.py train C N | frame SRC TGT | france` → `data/work/p65/sm/` (S3 `amc26-p65/data/`). Each run ≈ 2 min, ≤ 1.3 GB. Band-only copies are in `sm_band/`.
- `solution_p65.py`: the SageMaker job, shipped with `sm_launch.py --solution`.
  - Train, save to `results/<job>/model/`, then decode `params['decode']` frames in chunks → `results/<job>/dec/`.
  - `train: []` plus `weights_s3: results/<job>/model` = decode only.
  - `python solution_p65.py` runs a tiny-T5 CPU self-check.
- `ev65.py`:
  - `fetch JOB`;
  - `feats JOB FRAME` (≤ 1.0 GB; 10–30 s per frame, chunked);
  - `loco US_JOB IN_JOB [tag]` (`P65_DIRS`, `P65_OOD=1`; ≈ 1–3 min, ≤ 0.96 GB);
  - `frscore POOL US IN [fs]` (France combiner scores + `data/work/p55/cf_p65<fs>.npy` link for `p55.py build … mode=p65<fs>`; built, never run);
  - `self`.
- `iu65.py feats C JOB | eval`: the India/US arm on P57's full holdout frame (`P57_THREADS=4`). Feats ≈ 30 s / 1.2 GB per country; eval ≈ 4.5 min / 1.32 GB.
- **Local work files:** only the feature tables in `data/work/p65/` (32 MB) remain. The frames and decodes were deleted locally to free disk (1.3 GB left); they are on S3 under `amc26-p65/data*/` and `amc26-p65/results/<job>/dec/`. `ev65.py fetch <job>` re-syncs the decodes, and `prep65.py` rebuilds the frames in ≈ 2 min each.
- Trained correctors (bf16 safetensors, 599 MB each) are on S3: `s3://sagemaker-114872709824-ap-south-1/amc26-p65/results/{p65-locoUS-0927-130626, p65-locoIndia-0927-131232}/model/`. The pooled job was stopped before it saved anything usable for France.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- google/byt5-small: HF card licence `apache-2.0`, 299.6M params (counted in the job). Weights were pulled on AWS only (HF → S3 cache `amc26-p65/weights/byt5-small`).
- Training data = labelled train pairs only. No external data, and no model > 8B anywhere.
- **Worktree:** I worked in the main checkout's `probes/P65-byt5-corrector/`, since disk was 2.2 GB free (like P42 / P63). Nothing under `src/` was touched, and nothing is committed.
- **Timing:** the France LOCO verdict landed at ~14:20, not 14:00, and the India/US verdict at 14:47. Causes: GPU capacity waits (all types pending ~8 min), the .bin-filter relaunch, and memguard pausing the local eval.

## Open questions   (for the master; each with your best guess)
- Would a France-in-domain denoiser (France S1 + synthetic noise) work? Best guess: it is the only way this signal reaches France. It would take ≈ 2.5 h (generate, train, decode, and a LOCO analogue with US-S1-synthetic → US), too late for today's round.
