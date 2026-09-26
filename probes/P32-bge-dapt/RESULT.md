# P32-bge-dapt — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **LOCO gate: FAIL under the brief's rule (≥ +0.0005 in both directions).** The measure is DAPT vs a no-DAPT control fine-tuned on the same instance, with P13's France procedure.
  - **US→India (MLM on India text): +0.00210 ± 0.00018.** AUC(r) on the India band goes 0.818 → 0.896.
  - **India→US (MLM on US text): −0.00017 ± 0.00014**, i.e. neutral.
  - Run-to-run noise of the fine-tune is ≤ 0.00034.
  - DAPT helps only when the target's vocabulary is foreign to the fine-tune data (romanized Indian names for a US-trained CE). It never costs more than 0.0002.
- **France is not adopted (per the brief).** The France chain ran anyway ($7.74) and is delivered as an unvalidated one-lever spare:
  - **`output/sub32` = #31 (LB 0.9856) with only the France bge r replaced by the DAPT model's.** Validator PASS.
  - It differs from #31 in 6,119 France S1 (+2,725 / −3,520 pairs). India/US and `candidate_pairs.tsv` are byte-identical to #31.
  - France band CE-vs-GBDT disagreement drops from 33.5% to 25.2%, a label-free hint in DAPT's favour.
- **Kit:** `gdrive:amc26-offload/dapt_kit/p32_dapt_kit.zip`, **272.2 MB**, md5 `d95db65e383dfbe2963540eaffbb4c02` (Drive md5 verified). `bash run.sh` runs one chain per GPU.
- **AWS spend: $14.33.**

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **Default: adopt nothing.** The gate fails in the India→US direction: −0.00017 ± 0.00014 (brief: ≥ +0.0005 in both).
   - India/US are unaffected either way: their CE is trained in-domain.
   - The France-DAPT pooled model is also neutral in-domain: −0.00007 ± 0.00005 vs P13's pooled bge on the #7 yardstick.
2. **Optional spare LB slot: `output/sub32`**, in this worktree (`/home/harsh1/github-repos/amc-26-wt/p32-bge-dapt/output/sub32/`, two TSVs; add code.zip). Validator PASS.
   - **What changes:** #31's France pipeline exactly (P31: c_hn on P29's rebuild → bge blend → floored tier → bxneg → cap), except for the r in the bge blend:
     - the r is `data/work/p32/r/r_daptFrance_pool_test_France.parquet` (DAPT on France text → P13's pooled fine-tune);
     - the weights are P13's France vector [0.9245, 0.2595, 0.0515], unchanged;
     - t is **0.9873**, which gives #9ab's 6.71% empty on the blended file before the rules (P31's convention; #31 used 0.9868).
     - After the rules: 6.479% empty, 3.114 matches per S1 (#31: 6.458% / 3.117). Inside P6's bands.
   - **Why it is worth one slot:** the risk is lopsided.
     - With P13's vector (fr_fix), the protocol this file uses, DAPT − control is **+0.00169 ± 0.00017 / +0.00026 ± 0.00011**: positive in both directions, but under the +0.0005 bar in one.
     - The worst proxy outcome under any protocol is −0.00017.
     - France's band disagreement drop (−8.3 pt) is larger than India's (−5.2 pt, where DAPT gained) and US's (−2.9 pt, neutral).
   - **Reading the LB:** France is ~15% of the LB, and the change touches 2.4% of France S1. Expect |Δ| ≲ 0.0005. Above #31 ⇒ also apply the DAPT r on top of #34 / #34_veto.
   - **To put the DAPT France on another India/US base:**
     `.venv/bin/python probes/P14-france-shift/combine.py <base_dir> data/work/p32/fr32_t9873.parquet 0.9873 <out>`
     - #34 keeps #31's France, so `<base_dir> = output/sub34` works as is.
     - The veto variant needs P30's veto re-run on the DAPT r. The DAPT r covers every pair of the file, including the p ≥ 0.98 pairs.
3. **Kit** (`gdrive:amc26-offload/dapt_kit/p32_dapt_kit.zip`): with the gate failed, run it only as an independent replicate. Use `bash run.sh loco`, ~30 min on 4 H100s, or `bash run.sh` for everything.
   - The zip's README time table is the earlier, conservative estimate (~2.1 h on 1 H100). The repo copy `probes/P32-bge-dapt/kit/README.md` has the measured-speed version (~1.7 h on 1, ~45 min on 3).
   - Both are otherwise identical; the scripts and data did not change after the upload.

## Evidence         (table: ledger name | world | key metrics | verdict)
**LOCO gate** (`gate-daptIndia_locoUS_holdout+daptUS_locoIndia_holdout`, the later of the two rows with that name; world holdout-dense-half1):
- Families:
  - **dapt** = MLM on the target country's text (2 epochs × 1.62M strings), then P13's LOCO fine-tune (1 epoch) on the source country;
  - **ctl** = the same fine-tune without MLM, on the same instance, right after;
  - **p13** = P13's own LOCO bge (A10G, yesterday).
- Gains are vs m2_big without a CE under the same procedure. Differences are paired per S1 on the target's report half 1 (± SE).

| protocol | US→India: dapt / ctl / p13 gain | **dapt − ctl** | dapt − p13 | ctl − p13 (noise) | India→US: dapt / ctl / p13 gain | **dapt − ctl** | dapt − p13 | ctl − p13 |
|---|---|---|---|---|---|---|---|---|
| **fr_own**: France procedure (each family's mean OOD vector, shipped rule, P6 label-free t) | +0.00455 / +0.00245 / +0.00280 | **+0.00210 ± 0.00018** | +0.00175 ± 0.00018 | −0.00034 ± 0.00012 | +0.00131 / +0.00147 / +0.00148 | **−0.00017 ± 0.00014** | −0.00018 ± 0.00014 | −0.00001 ± 0.00008 |
| fr_fix: same procedure with P13's vector [0.9245, 0.2595, 0.0515] (the vector `sub32` uses) | | +0.00169 ± 0.00017 | +0.00130 ± 0.00016 | | | +0.00026 ± 0.00011 | +0.00022 ± 0.00011 | |
| xw: cross-fit OOD weights, D3 on target half 0 (no target labels) | | +0.00164 ± 0.00017 | +0.00140 ± 0.00017 | −0.00025 ± 0.00010 | | −0.00004 ± 0.00015 | −0.00022 ± 0.00016 | −0.00019 ± 0.00010 |

| family | target band AUC(r) US→India / India→US | stacked AUC (p+r) | source in-domain AUC(r) | own OOD vector w_own | label-free t on target | band CE-vs-GBDT disagreement (target) |
|---|---|---|---|---|---|---|
| dapt | **0.8957** / 0.8391 | 0.9596 / 0.9332 | US 0.9531 / India 0.9676 | [0.8778, 0.3679, −0.0413] | 0.77 / 0.67 | 14.5% / 22.9% |
| ctl | 0.8177 / 0.8156 | 0.9479 / 0.9304 | US 0.9526 / India 0.9678 | [0.9294, 0.2439, 0.0680] | 0.73 / 0.72 | 19.7% / 25.8% |
| p13 | 0.8209 / 0.8290 | 0.9488 / 0.9318 | US 0.9525 / India 0.9675 | [0.9241, 0.2595, 0.0516] | 0.71 / 0.70 | 20.0% / 26.1% |

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| gate-daptIndia_locoUS_holdout+daptUS_locoIndia_holdout | holdout-dense-half1 | tables above | **drop** (gate: both directions ≥ +0.0005) |
| gate-bge_locoUS_mid_holdout+bge_locoIndia_mid_holdout | holdout-dense-half1 | **evaluator self-check** on P13's files. The p13 family reproduces P13's published numbers: xw +0.00271 / +0.00133, France procedure +0.0028 / +0.00148, w_own [0.9241, 0.2595, 0.0516] (P13: [0.9245, 0.2595, 0.0515]) | self-check ok |
| (job logs) MLM | aws | Masked-token loss / accuracy on a fixed sample, after the head warm-up → after 2 epochs (12,974 steps, bs 256, lr 5e-5): India 6.02 / 20.3% → 1.68 / 69.2%; US 6.61 / 15.4% → 1.78 / 66.4%; **France 5.08 / 31.9% → 0.96 / 80.6%**. Speed: 1,234–1,547 texts/s on an L40S, 598 on an A10G | – |
| pool-daptFrance_pool_holdout_vs_bge_pool_holdout_plain | holdout-dense-half1 | **In-domain** (#7 yardstick, shipped rule): France-DAPT pooled bge vs P13's pooled bge **−0.00007 ± 0.00005** (402 S1 up / 272 down). Holdout AUC(r): India 0.9711 / US 0.9563 (P13: 0.971 / 0.956) | neutral (no in-domain cost) |
| (France band, label-free) | test-France | On P31's band (319,752 pairs), DAPT r vs P13 r: logit corr 0.761; CE-vs-GBDT disagreement **33.5% → 25.2%**; r ≥ 0.5 on 33.6% → 23.9% of pairs; 19.1% of band pairs cross 0.5 | – |
| frfile-chn_fix_dapt | test-France | P31's blended file with the DAPT r, P13's vector. t 0.9873 → 6.707% empty, 3.060 per S1. vs P31's blend at 0.9868: +6,038 / −5,189 pairs, 10,800 S1 changed | keep (spare) |
| frfile-chn_fix_dapt_wown | test-France | the same with the DAPT family's own vector [0.8778, 0.3679, −0.0413]: t 0.9888, +9,291 / −9,733 pairs, 17,971 S1 changed. Not staged (fr_own is the protocol that was negative India→US) | not staged |
| p32-sub32 | test | P31 `post` at t 0.9873: tier +4,735, bxneg +9,352, cap −17 → 6.479% empty, 3.114 per S1, in P6's bands. `combine.py` on #8c → **validator PASS**. **vs uploaded #31: 6,119 France S1 differ (+2,725 / −3,520 pairs); every non-France row identical; `candidate_pairs.tsv` md5-identical** | spare |
| frfile-selfcheck_p13r | test-France | builder check: P13's r on P19's file = #9ab exactly (0 pairs changed, t 0.985, 6.71%, 3.018 per S1). On P31's blended file with the old r: t 0.9868, 0 changes, and P31's `post` = `fr31_t9868` exactly (max \|Δp\| 0) | self-check ok |
| p32-kit-upload | – | `gdrive:amc26-offload/dapt_kit/p32_dapt_kit.zip`, 272.2 MB, md5 d95db65e… (Drive md5 = local). Rehearsed end to end on a tiny random XLM-R with the real data slices: 1-GPU sequential, and 2 "GPUs" in parallel (all 6 r files validated) | keep |
| p32-sm-launch / p32-sm-cost | aws | `p32-a-0926-1729` ml.g6e.xlarge 4,109 s = $3.57; `p32-b-0926-1729` ml.g6e.2xlarge 3,233 s = $3.02; `p32-fr-0926-1729` ml.g5.4xlarge 11,422 s = $7.74. **Total $14.33** (on-demand Pricing API rates; S3 ≈ 2.5 GB of checkpoints, cents) | – |

## What failed      (approaches tried and dropped, with the number that killed them)
- **DAPT when the target is US (India→US): −0.00017 ± 0.00014** (fr_own, vs the same-instance control); fr_fix +0.00026 ± 0.00011; xw −0.00004 ± 0.00015.
  - AUC(r) on the US band is +0.024 over the control, but the control itself is 0.013 below P13's identical run, so AUC swings between runs are large. The F0.5 effect is nil.
  - This direction fails the brief's gate.
- **The DAPT family's own OOD vector for France.** It puts 0.37 on logit r vs P13's 0.26. It helps US→India (fr_own +0.00210 vs fr_fix +0.00169) but is the only protocol with a negative India→US. Not used in `sub32`.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`solution_p32.py`** is the SageMaker job and also the kit script.
  - It is shipped as `solution.py` with P13's `solution_p13.py` next to it, and reuses P13's `Job` unchanged: load, tokens, train, holdout, test, save/restore.
  - `main(train_dir, test_dir, model_dir, out_dir, params)`. `params['chains']` is a list of `{dapt, ft, tag, tests, save_ft, save_dapt, epochs}`, each run on a fresh bge. Locally: `python solution_p32.py DATA OUT '<json>'`.
  - `mlm(job, tok, sc, name)` runs the MLM on `dapt_<c>.parquet` (column `t`):
    - 15% masking (80/10/10), max_len 64, frozen word embeddings tied to a fresh LM head;
    - 300 head-only steps, then encoder lr 5e-5 (6% warm-up, linear decay) and head lr 5e-4;
    - bs 256, bf16, OOM-halving micro-batches;
    - `dapt_eval` = masked-token loss/accuracy on a fixed 4,096-text sample with a fixed mask.
  - Cards ≥ 40 GB get micro 128 / 256 and scoring bs 2,048; the per-example recipe is unchanged.
  - Speed and memory:
    - MLM: 1,234–1,547 texts/s on an L40S, 598 on an A10G; peak GPU memory 28.8 GB (L40S) / 16.2 GB (A10G).
    - Fine-tune (P13's): 762–970 pairs/s on an L40S, 337 on an A10G. Scoring: 2,800–4,700 / 1,420–1,940 pairs/s.
    - Wall time: a LOCO chain takes 43–53 min on an L40S; the France chain 3.1 h on an A10G.
  - Outputs:
    - `r/r_<tag>_holdout.parquet` and `r/r_<tag>_test_France.parquet`, as (s1, sx, r) in `data/work/p13/r` format;
    - `ckpt/ce_dapt<c>.pt` and `ckpt/ce_<tag>.pt` in P13's format.
  - Checkpoints, in `s3://sagemaker-114872709824-ap-south-1/amc26-p32/results/<job>/ckpt/`:
    - `ce_daptIndia.pt` and `ce_daptUS.pt` (jobs a / b);
    - `ce_daptFrance.pt` and **`ce_daptFrance_pool.pt`** (the France model, `p32-fr-0926-1729`).
- **`p32.py`** is the laptop side:
  - `build`: MLM corpora + France union band, 28 s, 1.9 GB. The France union is P13's m2_big band ∪ #9ab/P26 ∪ P29/P31 keys = 1,447,625 pairs; P13's texts are asserted equal.
  - `tiny` / `bundle`: the rehearsal model and source bundle. `launch` / `fetch` / `collect`: the jobs.
  - `gate [tagUS tagIndia ctlUS ctlIndia]`: 3 families × 3 protocols, paired per-S1 differences. 100 s, 1.4 GB.
  - `pool <tag> [ctl] [rule]`: P13's in-domain yardstick, ledger row under P32. ~10 min, 2 GB.
  - `frfile src name r_file [w]`: a shipped France file (s1, sx, p, p_gbdt) with its 0.02–0.98 band re-blended with the given r. It prints the t curve, picks the grid t (step 0.0001) closest to #9ab's 6.71% empty, and reports the churn. 3 s, 0.7 GB.
  - `kit`: stage + zip the kit.
- **France r (s1, sx, r):** `data/work/p32/r/r_daptFrance_pool_test_France.parquet`, 1,447,625 pairs. It covers every pair of P29/P31's file, of #9ab/P26's file, and of P13's m2_big band.
- **Files:**
  - `data/work/p32/prob_test_France_chn_fix_dapt.parquet`: the blended file before the rules.
  - `data/work/p32/fr32_t9873.parquet`: after P31's post-rules.
  - `output/sub32/`: the staged submission.
- **Rebuild `sub32`** (from this worktree):
```
W=/home/harsh1/github-repos/amc-26-wt/p32-bge-dapt
.venv/bin/python probes/P32-bge-dapt/p32.py frfile data/work/p31/prob_test_France_chn_fix_bge.parquet chn_fix_dapt data/work/p32/r/r_daptFrance_pool_test_France.parquet 0.9245,0.2595,0.0515
(cd ../p31-final-france && .venv/bin/python probes/P31-final-france/p31.py post $W/data/work/p32/prob_test_France_chn_fix_dapt.parquet 0.9873 $W/data/work/p32/fr32_t9873.parquet)
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p32/fr32_t9873.parquet 0.9873 output/sub32
cd data/student_resource && python3 utils/validate_submission.py --matching $W/output/sub32/matching_results.tsv --candidate $W/output/sub32/candidate_pairs.tsv --test-dir dataset/test
```
- **`kit/`**: `README.md`, `requirements.txt` (pinned to the DLC), and `run.sh`: `bash run.sh [all|loco|france] [--gpus N]`, one process per GPU (round-robin, longest first), logs in `logs/<chain>.log`, r files checked at the end.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / data.** The model is still bge-reranker-v2-m3 (Apache-2.0, 568M params). The MLM uses only unlabelled test (France) or train (India/US) records, which the organiser ruling allows ("unlabelled test data may be used"). No external data.
- **`sub32` is unvalidated for France.** The proxy gate failed in one direction.
  - Its logit correlation with P13's r on the France band is only 0.76, so this is a substantially different CE.
  - The post-rules (tier, bxneg) mask most of that: 6,119 S1 change vs #31, against 10,800 before the rules.
- **The proxy corpus is an assumption.** The India/US MLM corpora mimic France's: every scored (holdout-band) record, plus random records of the same country, up to France's 1,622,480 strings.
- **The OOD weight vector** (fr_own) averages the two cross-fits, as P13's France vector does, so each direction's vector contains one fit on that target's half 0. The same leak applies to every family. `xw` is the leak-free check.
- **Hardware mix.** The LOCO chains ran on L40S and the France chain on A10G, all with bf16. The controls ran on the same instances as their DAPT chains, so the gate compares like with like.
- The ledger has two rows named `gate-daptIndia…`. The first lacks the control family, which had not landed yet; the later one is the verdict.

## Open questions   (for the master; each with your best guess)
- **Does France behave like the India target (gain) or the US target (neutral)?**
  - Label-free hints:
    - The base model already predicts French best: masked-token loss after the head warm-up is France 5.08 vs India 6.02 and US 6.61.
    - DAPT cut France's band disagreement by 8.3 pt, more than India's 5.2 (gain) and US's 2.9 (neutral).
  - Best guess: a small positive. Only the LB can tell, and `sub32` isolates it against #31.
- **Why only India?** The P3 texts are ASCII-folded (0% non-ASCII, even for India), so it is not Indic script.
  - Best guess: the US-trained CE had never seen romanized Indian names and localities, and MLM on them helps it place those tokens.
  - An India-trained CE already knows the English-like US vocabulary.
- **Worth a France-DAPT veto (P30's rule with the DAPT r)?** Only if `sub32` beats #31. The DAPT r already covers all the confident pairs.
