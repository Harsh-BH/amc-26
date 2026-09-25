# P4b-rerank — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The answer:** a multilingual-e5-small cross-encoder reads both records' "name | addr" and rescores P4-M2's uncertain band. It catches the look-alike street mismatches that the features miss: band AUC 0.79 → 0.93, look-alike AUC 0.82 → 0.98.
- **The number:** full-corpus holdout (half 1, D3 re-tuned on half 0) goes from **0.9815 → 0.9881 (+0.0066 ± 0.0002)** with band 0.02<p<0.98. With the cheaper 0.1<p<0.9 band it is +0.0055.
- **Recommendation:** adopt it for India and US. Test inference is about 20 min on the GPU.
  - It does **not** transfer to a country it never saw: leave-one-country-out loses up to −0.0016.
  - So France, or any label without training data, keeps P4's p unchanged. Expected test gain ≈ +0.006.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**What to promote.**
- `probes/P4b-rerank/rerank.py` → `src/rerank.py`.
- The fine-tuned weights `data/work/p4b/ce.pt`: 43 MB fp16, every weight except the frozen word embeddings. The base model is `intfloat/multilingual-e5-small`, from the HF cache.

**Pipeline step.** It sits between the matcher and the decision layer, per country, in **its own process**. torch + CUDA alone is 1.4 GB RSS, so it can't live inside `run.py`'s cap.
```
flock /tmp/amc-gpu.lock .venv/bin/python src/rerank.py rescore <split> <country> in.parquet out.parquet
```
- `in.parquet` has columns `[s1, sx, p]` (P4 probabilities, entity ids).
- `out.parquet` has the same rows with `p` replaced by p′.
- In-process equivalent: `apply(split, country, s1, sx, p, CFG["w"], band=CFG["band"]) -> p′`.
- Config: `CFG = {"band": (0.02, 0.98), "w": [0.636, 0.757, 0.355], "countries": ("India", "US")}`.

**The blend rule.** For pairs with 0.02 < p < 0.98 whose country is in `countries`:
- p′ = σ(0.636·logit p + 0.757·logit r + 0.355), where r = σ(CE logit).
- Every other pair keeps p.
- The weights are a logistic stack fitted on holdout half-0 band pairs.

**Re-tune the decision on p′, not on p.**
- D3 moves from t 0.71 / t0 0.72 to **t 0.70 / t0 0.73**.
- For P5's final rule (D4/D7…), `rerank.py blend_file` wrote `data/work/p4b/prob_holdout_blend.parquet`. It has P4's `prob_holdout` schema plus a `p_gbdt` column, so P5's `bench.py` can run on it with a one-line SRC entry.

**Budget fallback.** If test time is tight, use band (0.1, 0.9) with w = [0.631, 0.784, 0.329]. That is +0.0055, 2.5% of pairs and about 8 min.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| ce-e5small-train-v1 | train fold-3 band (40k) | band AUC: fold-2 LightGBM 0.809, CE **0.926**, sum 0.928. 778k train pairs, 538 pairs/s, 25 min, 2.4 GB VRAM | keep |
| p4-D3-base | holdout-dense-half1 | F0.5 0.9815 (India 0.9803, US 0.9822), singleton acc 0.976, P 0.9941 / R 0.9588. False positives on look-alikes (same name core, house number on both sides): 730 | reference |
| p4-D3-blend-0.1-0.9 | holdout-dense-half1 | F0.5 **0.9869 (+0.0055)**, singleton 0.985, P 0.9965 / R 0.9684, look-alike FP 475. Band 77k pairs (2.5%): AUC p 0.792 / r 0.914 / blend 0.931; look-alike AUC 0.823 / 0.978 / 0.979 | keep |
| p4-D3-blend-0.05-0.95 | holdout-dense-half1 | F0.5 0.9876 (+0.0061), band 3.8%, look-alike FP 406 | keep |
| p4-D3-blend-0.02-0.98 | holdout-dense-half1 | F0.5 **0.9881 (+0.0066)**: India 0.9888 (+0.0085), US 0.9876 (+0.0054). Singleton 0.988, P 0.9975 / R 0.9694, look-alike FP **340**, band 6.2% | **keep (adopted)** |
| p4-D3-blend-0.2-0.8 | holdout-dense-half1 | F0.5 0.9855 (+0.0040), band 1.4% | keep (narrower = less) |
| p4-robust-se-loco | holdout-dense-half1 | Paired s.e. of the gain 0.0002 for both bands: 4,866 S1 up / 605 down (0.1–0.9), 5,626 / 657 (0.02–0.98). **LOCO** band 0.1–0.9, results below | the gain is real; it is country-specific |
| france-labelfree-rule-agreement | test France + holdout | On rules-accepted pairs, share with r>0.5: India 0.940 / US 0.890, = the rule precision 0.943 / 0.893. In-domain the CE passes 99.5% of true and 1.3% of false pairs; LOCO CEs pass 30–45% of false. France: 0.851, consistent with either behaviour | inconclusive |
| rescore-test-dryrun-cost | test India, 100k pairs | 128 s end to end (id index + texts + scoring), 2.27 GB peak. Holdout `rescore` equals the cached r exactly (max \|Δp\| 0.0); France returned unchanged | keep |
| p5-D3-base / p5-D3-blend-* | holdout-dense-half1 | On P5's weaker proxy: 0.9157 → 0.9641 (0.1–0.9) / 0.9750 (0.02–0.98) | proxy only |

LOCO results (from p4-robust-se-loco):

| train → test | base | blend, weights fit on the train country | blend, weights fit on the test country | CE trained on both countries | band AUC, p vs LOCO r |
|---|---|---|---|---|---|
| India → US | 0.9822 | 0.9806 (**−0.0016**) | 0.9834 | 0.9868 | 0.789 vs **0.713** |
| US → India | 0.9802 | 0.9801 (−0.0001) | 0.9817 | 0.9872 | 0.791 vs **0.721** |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Transfer to an unseen country.** A CE trained on one country scores the other country's band *worse* than P4's p (AUC 0.71–0.72 vs 0.79). Blending it in loses up to −0.0016. Hence the `countries` gate.
- **max_len 96, unbucketed batches.** 311 train pairs/s, because batches pad to 92 of 96 tokens. max_len 64 + length buckets gives 540 pairs/s; longest_first truncation only drops the address tail (city/state).
- **Narrow band 0.2–0.8.** +0.0040 against +0.0066 for 0.02–0.98: most of the gain sits in pairs P4 already calls fairly confident.
- **Not tried:** xlm-roberta-base / muril (disk had 1.8 GB free at the start, and e5-small was already cached), more than 1 epoch, raw (un-normalised) text.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
`probes/P4b-rerank/rerank.py`. The docstring lists every command.

**Pipeline and blend**
- `rescore(split, country, in, out)`: the pipeline step. ~780 pairs/s end to end on test (1,200–1,700 pairs/s scoring only), 2.3 GB peak.
- `apply(split, country, s1, sx, p, w, band) -> p′`.
- `blend(p, r, w, lo, hi)`.
- `fit_blend(p, r, y) -> [a, b, c]`.

**Model and texts**
- `score(ta, tb, tok, m) -> r`: length-sorted, bf16, batch 512.
- `load_model(path)`.
- `texts(split, country, q1, qx)`: one SX text file in RAM at a time.
- `rows(split, country, s1, sx) -> (q1, qx)`: entity ids → rows of P3's texts.

**Training**
- `band(country)`: LightGBM on P4 X, fit on fold 2; writes `trainband_<c>.parquet`. 70 s India / 195 s US, 2.0 GB.
- `train(countries, tag)`: 25 min for both countries, 2.64 GB RSS, 2.4 GB VRAM.

**Evaluation**
- `evaluate(src)`, `robust(src)`, `france_check()`, `d3()`.
- `d3()` is an exact, fast re-implementation of P5's D3 sweep (identical optimum 0.9157 t .65 / t0 .60 on the proxy).

**Full-test cost.** The band is 6.2% of India + US pairs if P4's test p looks like its holdout p. That is about 1.44M pairs → **~20 min** (0.1–0.9: 0.57M pairs, ~8 min). France costs 0 (gated). Fits the 1 h budget.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences.**
  - Model: intfloat/multilingual-e5-small, MIT, 118M params (research_licences.md), used offline from pretrained weights.
  - Libraries: torch (BSD), transformers / tokenizers / safetensors (Apache-2.0), scikit-learn (BSD).
  - Pin the versions used here: torch 2.14.0, transformers 5.17.0. No external data, no API.
- **France / unseen labels.**
  - The gate `countries = ("India", "US")` means "the CE's training countries". It is not a row filter: France rows get P4's p and go through the same decision.
  - France gets no reranker gain. Applying the CE there would likely *lose*: LOCO −0.0016, and the stack weights are nearly the same as the LOCO ones.
  - The label-free France check can't tell the two cases apart.
- **GPU dependency at test time.** On CPU, e5-small would be several times slower. The 0.02–0.98 band could exceed 1 h there; use the 0.1–0.9 fallback.
- **Band definition.**
  - Weights and band were fit on P4-M2's holdout p. If P4 retrains, the band shifts: refit `fit_blend` on the new holdout (2 min, CPU; `eval p4` does it).
  - r only needs rescoring for new band pairs (`score p4 band` rescores only missing pairs).
- **Holdout optimism.** Holdout pairs share the train S2/S3 corpus with the CE's training pairs. Any memorisation of an SX's label from another S1 pushes holdout pairs the *wrong* way, so the estimate is, if anything, conservative. On test, no records are shared.
- **Decision interplay.**
  - The gain was measured with D3. P5's D7 stacking re-derives p from context, and the two gains may overlap.
  - The master should run P5's final rule on `prob_holdout_blend.parquet` against `prob_holdout.parquet` before adopting both.
- **Assumption.** The train band came from a fold-2 LightGBM on P4's X, because P4 writes no out-of-fold train probabilities. It is weaker than M2 (fold-3 band AUC 0.81), but the evaluation used M2's real band.

## Open questions   (for the master; each with your best guess)
- **Does the fine-tuned `ce.pt` (43 MB) go into the code zip, or is it retrained from `train` (25 min GPU)?** Best guess: ship it, plus the training command in README, since the base weights come from HF.
- **May a France-specific reranker be fit on test pseudo-labels** (P4-confident France pairs as labels)? That is PROBLEM.md Q4. Best guess: skip. Confident pairs don't teach the band, and fitting on test is unclear under the rules.
- **Band 0.02–0.98 or 0.1–0.9?** Best guess: 0.02–0.98 if the GPU is available at submission time, else 0.1–0.9. The difference is +0.0011 for about 12 extra minutes.
- **Does the gain survive P5's final decision rule?** Best guess: mostly yes. The CE's information is textual, and D7 only sees p's context; about +0.004 to +0.006 should remain.
