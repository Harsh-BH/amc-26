# P29-fr-tiers — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
The France rebuild (#9ab's IDF world + '(n)' house-number fix + R1 fillers + a bge top-up, **$0.11**) and the key tier are both staged and pass the validator. India/US rows are byte-identical to #8c.
**The key tier as briefed loses in-country** (US −0.00023 ± 0.00004, India −0.00164 ± 0.00014). With a **p ≥ 0.5 floor** it is neutral at #8c's t, and it gains **+0.00022 ± 0.00005 (US) / +0.00021 ± 0.00005 (India)** on the France-like proxy (t 0.985), where the rescued pairs are 96–97% true.
**Upload `output/sub29b`** (rebuild + floored tier + cap, t 0.9875: empty 6.63%, 3.06 matches per S1). `sub29a` (rebuild only) isolates the rebuild. The per-S1 cap is harmless and almost never fires (2 holdout S1 per country, 14 France S1).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **LB order.** Upload `sub29b` first. It contains every lever, and the tier is the only one with label support.
  - If a second slot is free, use `sub29a` to split the rebuild from the tier.
  - Or use `sub29b_t983` (the 6.43%-empty t) to test whether France's t should come down.
  - All folders are in `/home/harsh1/github-repos/amc-26-wt/p29-fr-tiers/output/`. They hold only the two TSVs, so add code.zip.

| folder | France file (`data/work/p29/`) | t | France empty | matches / S1 | France S1 non-empty | validator |
|---|---|---|---|---|---|---|
| `sub29a` | `prob_test_France_fix_bge.parquet` | 0.9875 | 6.70% | 3.035 | 242,069 | PASS |
| `sub29b` | `prob_test_France_fix_bge_tier_t0.9875.parquet` | 0.9875 | 6.63% | 3.060 | 242,259 | PASS |
| `sub29b_t983` | `prob_test_France_fix_bge_tier_t0.983.parquet` | 0.983 | 6.40% | 3.119 | 242,846 | PASS |
| (#9ab, for reference) | p19 `prob_test_France_idf144_bge.parquet` | 0.985 | 6.71% | 3.018 | 242,044 | LB 0.984072 |

- **Threshold.** France t at #9ab's empty rate (6.71%) = **0.9875** (6.70%, 3.035 per S1). 3.02 matches per S1 would need t 0.988 (6.74%). The rebuild adds pairs to already-matched S1, so the two #9ab targets cannot both be hit.
  - **t for empty = 6.43% is 0.983** (6.434%, 3.104 per S1) on the rebuild alone. With the tier + cap at 0.983 the file sits at 6.40%.
- **src promotion (P9), if it wins:**
  1. `normalize.house_number` / `address_fields`: pre-apply `p29.paren_hn` (`(^|,\s*)\((\d+)\)` → `\1\2`) to the address. Footprint outside France S2/S3: 0 France S1, 0 US records, and India 296 train / 233 test records of all sources. Their effect on India was not measured; it is tiny, and the regex is general code.
  2. `normalize.GENERIC` += {fils, cie, associes, freres, frs} (R1, P20).
  3. bge must score the rebuilt band. That is 368,762 France pairs; the pipeline already scores the band.
  4. Post-rule in `run.py` after the blend, for every country:
     - `p29.tier_mask(k1, kx, chain, am) & (p >= 0.5)` → p := 1;
     - then `p29.cap_mask(s1, sx, p, pred)`. An SX over the cap loses every pair.
     - It is general code: no country key, and the chain size is counted per country over S1.
     - At #8c's own t it changes US/India by +0.000001 / −0.000004. Applying it to all countries is safe but pointless there.
- **Commands (reproduce):**
```
.venv/bin/python probes/P29-fr-tiers/p29.py rf                                   # 2 s: data/work/p29/w_fix (hn-fixed rf)
ER_WORK=data/work/p29/w_fix .venv/bin/python probes/P29-fr-tiers/p29.py rescore  # 200 s, 2.6 GB: P20 rescore 'noise'
.venv/bin/python probes/P29-fr-tiers/p29.py build                                # 9,248 new band pairs + 2,000 controls
.venv/bin/python probes/P29-fr-tiers/p29.py launch --profile harsh --region ap-south-1 --instance ml.g5.xlarge --no-spot \
  --prefix amc26-p29 --name p29-score-0926-1643 --data data/work/p29/sm_data --solution probes/P13-aws-rerank/solution_p13.py \
  --max-run 3600 --hp '{"phases": ["bge_ckpt"], "ckpt": "s3://sagemaker-114872709824-ap-south-1/amc26-p13/results/p13-rerank-0925-2257/ckpt/ce_bge_pool.pt",
  "s3": "s3://sagemaker-114872709824-ap-south-1/amc26-p29/results/p29-score-0926-1643", "deadline_s": 3000}'
.venv/bin/python probes/P29-fr-tiers/p29.py fetch p29-score-0926-1643 && .venv/bin/python probes/P29-fr-tiers/p29.py check
.venv/bin/python probes/P29-fr-tiers/p29.py blend                                # -> prob_test_France_fix_bge.parquet
.venv/bin/python probes/P29-fr-tiers/p29.py profile
.venv/bin/python probes/P29-fr-tiers/p29.py holdout                              # 130 s, 2.3 GB (P23's #8c dump)
.venv/bin/python probes/P29-fr-tiers/p29.py tiers data/work/p29/prob_test_France_fix_bge.parquet 0.9875 data/work/p29/prob_test_France_fix_bge_tier_t0.9875.parquet
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p29/prob_test_France_fix_bge.parquet 0.9875 output/sub29a
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p29/prob_test_France_fix_bge_tier_t0.9875.parquet 0.9875 output/sub29b
.venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p29/prob_test_France_fix_bge_tier_t0.983.parquet 0.983 output/sub29b_t983
cd data/student_resource && python3 utils/validate_submission.py --matching <out>/matching_results.tsv --candidate <out>/candidate_pairs.tsv --test-dir dataset/test
```

## Evidence         (table: ledger name | world | key metrics | verdict)
### 1. France rebuild
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p29-hn-paren-rf | test-France | 23,272 of 1,434,993 France S2/S3 (1.62%) open a segment with '(n)'. For 23,104 of them the hn was empty. 23,250 hn change, 0 end up empty. France S1: 0 rows. Unfixed recomputation == stored rf hn (asserted) | keep |
| p29-rescore-fix-noise | test-France | P14's idf144 X + hn-fixed rf + R1, re-derived through rules → features.strings → pruner → m2_big. 1,332,223 kept pairs (#9ab 1,305,341: 5.13 vs 5.03 per S1), 6,863 rule-only extra. **hn fix alone** (vs P20's IDF+R1 file, t 0.975): +615 pairs (100% hn-equal) / −498 (70.5% sibling offset SIB3), 1,062 S1 changed. 200 s, 2.59 GB | keep |
| p29-bge-build | test-France | band (0.02 < p < 0.98) 368,762. 359,514 already had a P13/P19 r; **9,248 new**. Control texts == P13's | keep |
| p29-sm-done | aws | `p29-score-0926-1643`, ml.g5.xlarge, **234 billable s = $0.11** | keep |
| p29-sm-repro | test+holdout | control r vs P13's r: corr ≥ 0.9999, median \|Δlogit\| 0, p99 0.06 (France 0.13, bf16 batch noise, as in P19) | keep |
| p29-blend-fix-bge | test-France | #9ab's recipe (`rr.blend(p, r, [0.9245, 0.2595, 0.0515], 0.02, 0.98)`). CE and GBDT disagree on 32.9% of the band | keep |
| profile-fix_bge | test-France | See the t curve below. **vs #9ab France** (0.985 vs 0.9875): +16,142 / −11,817 pairs, 26,267 S1 changed. Added pairs 94.7% hn-equal, 0.1% SIB3; dropped pairs 74.5% hn-equal, 3.3% SIB3, 15.6% hn missing. Same t 0.985: +17,772 / −2,643 | keep |

t curve of `prob_test_France_fix_bge.parquet` (France empty S1 / matches per S1):

| t | 0.975 | 0.98 | 0.983 | 0.985 | **0.9875** | 0.988 | 0.99 |
|---|---|---|---|---|---|---|---|
| empty | 6.28% | 6.32% | 6.43% | 6.54% | **6.70%** | 6.74% | 6.91% |
| per S1 | 3.150 | 3.138 | 3.104 | 3.076 | **3.035** | 3.025 | 2.984 |

### 2–3. Key tier and cap: US/India dense holdout (#8c's p / argmax / t from P23's dump, base F0.5 0.989899 = #8c)
Tier = the SX's argmax S1, with R1-cleaned name core, hn and city all equal and non-empty, and a core unique among the country's S1 (chain size 1). Δ = paired per-S1 F0.5 on P5's report half 1 ± SE.

| ledger name | arm | US | India | verdict |
|---|---|---|---|---|
| p29-tier-cap-holdout | tier pairs (holdout S1) / precision | 94,434 / 99.84% | 36,843 / 98.46% | — |
| | … already accepted by #8c | 94,315 | 36,296 | — |
| | **new pairs the tier forces / their precision** | 119 / **2.5%** | 547 / **0.5%** | — |
| | tier pairs with p < 0.5: precision | 0.85% | 0.37% | — |
| | tier as briefed, at #8c's t | **−0.000228 ± 0.000036** | **−0.001640 ± 0.000136** | **drop** |
| | tier with p ≥ 0.5 floor, at #8c's t | +0.000001 ± 0.000001 | −0.000004 ± 0.000004 | neutral |
| | France-like proxy: t 0.985; floored tier rescues pairs with 0.5 ≤ p < 0.985 | 274 pairs, 97.1% true | 191 pairs, 96.3% true | — |
| | **floored tier vs t 0.985 alone** | **+0.000221 ± 0.000045** | **+0.000210 ± 0.000054** | **keep** |
| | un-floored tier vs t 0.985 alone | −0.000009 ± 0.000057 | −0.001434 ± 0.000146 | drop |
| | cap (train max: 5 S2, 6 S3 per S1, checked on gt_pairs) | 2 S1, 2 pairs (0% true), +0.000000 | 2 S1, 2 pairs (0% true), +0.000002 | keep (harmless) |

- **Why the plain tier fails:** 116 of US's 119 forced pairs (544 of India's 547) are distractors that belong to no S1. They share name core, parsed hn and city. The model rejects them at p ≈ 0 because the parsed hn is only the leading digit run.
  - Schematic: '12-30 Main St' vs '12-31 Main St' both parse to 12. It is the same sibling offset, hidden in the second number group.
  - The floor lets the model veto those. They are almost all below p 0.1.
- **France looks different.** Only 80 of France's 253,211 tier pairs are below 0.5. The rejected ones sit at 0.7–0.98, just under France's high t. The in-country pairs at those p values are 96–97% true.
- **Applied to France** (`p29-tier-cap-France-t0.9875` / `-t0.983`), on the rebuilt file:

| t | tier pairs | ≥ floor | pairs added | empty S1 fixed | cap: S1 touched / pairs dropped | empty before → after | per S1 before → after |
|---|---|---|---|---|---|---|---|
| 0.9875 | 253,211 | 253,131 | **6,600** (100% hn-equal by construction) | **190** | 14 / 14 | 6.70% → 6.63% | 3.035 → 3.060 |
| 0.983 | 253,211 | 253,131 | 3,921 | 88 | 24 / 25 | 6.43% → 6.40% | 3.104 → 3.119 |

  - The brief expected ~7.8k pairs and ~267 empty S1. The rebuilt file already accepts more tier pairs than #9ab does. On #9ab's own file at 0.985, the un-floored tier would add 6,958 pairs and fix 192 S1.

### 4. Staged files
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p29-sub29a | test | validator PASS; India/US rows of both TSVs byte-identical to #8c (md5 over non-France lines) | keep |
| p29-sub29b | test | validator PASS; India/US identical to #8c; vs sub29a +6,600 / −14 pairs, 6,056 France S1 changed | keep → LB |
| p29-sub29b_t983 | test | validator PASS; India/US identical to #8c | keep (2nd t) |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Key tier without a p floor** (the brief's rule): −0.00023 US, −0.00164 India at #8c's t, and −0.0014 India even at t 0.985.
  - Precision is 99.8% / 98.5% over all tier pairs. But the model already accepts every true tier pair in-country. The only pairs the tier adds are the model's p ≈ 0 rejects, which are 0.5–2.5% true.
  - Not shipped. The floored version replaces it.
- Nothing else was dropped. Session note: the first session hit the usage limit at ~16:39. The rescore had already finished, and the AWS job was launched after the resume.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
`probes/P29-fr-tiers/p29.py` (`selftest` asserts the parser, tier, cap and F0.5 helpers):

- **Rebuild:**
  - `paren_hn(addr) -> addr`: the regex fix.
  - `rf()`: writes `data/work/p29/w_fix/` (symlinks to P14's `w_idfx` plus a new `rf_test_sx_France.parquet`). 2 s, 1.2 GB.
  - `rescore()`: P20's `rescore.main("noise")` with OUT = `data/work/p29/`. 200 s, 2.6 GB.
- **bge top-up:**
  - `build()`, `launch(argv)`, `fetch(job)`, `check()`, `blend()` follow P19's recipe. The r sources are P13 ∪ P19 ∪ this job (`data/work/p29/r/`).
  - Each step: seconds, < 1 GB.
- **Tier and cap:**
  - `tier_mask(k1, kx, chain, am) -> bool[]`: the key tier; the floor is applied by the caller.
  - `cap_mask(s1, sx, p, pred) -> bool[]`: the pairs to drop; `CAP = {"S2": 5, "S3": 6}`.
  - `keys(rf_path, ids) -> id → core (R1), hn, city`.
  - `tiers(file, t, out, floor=0.5)`: France post-rule → (s1, sx, p) for combine.py. 5 s, 1.1 GB.
  - `holdout()`: the US/India measurement. 130 s, 2.3 GB.
- **Profiling:**
  - `curve(d, ts)`, `t_at(c, empty)`, `profile(file)`: the t curve.
  - `diff(A:t, B:t)`: churn plus the hn lens.
- **Outputs in `data/work/p29/`:**
  - `prob_test_France_w_fix_noise.parquet` (m2_big only, 29 MB);
  - `prob_test_France_fix_bge.parquet` (s1, sx, p, p_gbdt; 34 MB);
  - `prob_test_France_fix_bge_tier_t{0.9875,0.983}.parquet`;
  - `r/`, `sm_data/`.
- Staged: `output/sub29a`, `sub29b`, `sub29b_t983` (217 MB each; the disk has ~3 GB free, so delete them after upload).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Threshold choice.** Matching #9ab's empty rate forces t up to 0.9875 and drops 11.8k of #9ab's pairs, 74.5% of them hn-equal. The step from 0.983 to 0.9875 drops 17.9k pairs, 78% hn-equal and 1.4% SIB3.
  - The label-free lens says a lower t (0.983, `sub29b_t983`) keeps more true-like pairs. I staged 0.9875 because it is the briefed profile rule.
  - If sub29b wins, try `sub29b_t983` next.
- **The tier's proxy is in-country.** The rescued-pair precision (96–97%) comes from US/India pairs with 0.5 ≤ p < 0.985. France's rescued pairs have the same keys and similar p (0.7–0.98), but France's m2_big is out of domain.
  - Expected France effect if the precision holds: roughly +0.2–0.3 pt France F0.5, about +0.0003–0.0004 LB (scaled per rescued pair from the US proxy). This is a rough estimate.
- **hn prefix weakness.** The tier's hn key is the leading digit run. In France, "12 bis" / "12-14" addresses parse to 12. The floor covers this in-country: the US/India false tier pairs are all below 0.1.
  - A stricter key (all digit runs equal) would be safer, but I did not test it.
- **Not included:** P15's sibling rule (R2). #9ab doesn't carry it either, as briefed. The hn fix makes R2 slightly stronger, since it exposes 70%-SIB3 pairs.
- **Candidates:** 1,332,223 France pairs (5.13 per S1, +2% vs #9ab). The extra pairs are the rule-only pairs that R1 and the hn fix create.
- **Compliance:** no new model or data. bge-reranker-v2-m3 is Apache-2.0 and 568M params. The regex and the filler list are hand-written general code. The chain size reads unlabelled test S1 (allowed, ruling 2).

## Open questions   (for the master; each with your best guess)
- **One slot: sub29b or sub29a?** Best guess: sub29b. The tier + cap is holdout-validated in the France-like proxy (+0.0002 in both countries), and the rebuild's lens is positive: added pairs 94.7% hn-equal, dropped pairs 74.5%.
- **France t:** 0.9875 (#9ab's empty rate) or 0.983 (6.43%)? Best guess: 0.983 if sub29b beats #9ab. The pairs dropped between the two are 78% hn-equal.
- **Promote the floored tier into src for all countries?** Yes, but only as a France-relevant rule. It is neutral where the model's t is well tuned, and it pays when t is set high (France). Cap: yes, it is free.
