# P3-block — RESULT
## TL;DR
The target is met: pair recall **0.9916 at 14.2 candidates/S1** (S1 full-recall 0.971), measured on holdout S1 against the FULL train corpus. That means all 10.3M S2/S3, with every S2/S3 searching the full 2.2M S1 pool (India 0.9906, US 0.9923).
It takes one sparse TF-IDF product per country, S2/S3 → S1: name char-3grams plus address words at 50/50 weight, within the SX's state group. Each SX keeps its top-2 S1 (top-10 if its address is empty), and each S1 keeps its top-10 among the SX that listed it.
Adopt `probes/P3-block/block.py` with its default `CFG`. **Test candidates are already built:** `data/work/p3/cands_test_{France,India,US}.parquet`. That is 27.4M pairs (15.8/S1), no duplicates, only S2-/S3- ids, and every test S1 covered except 1 India S1. It took 35 min on this laptop.

## Recommendation
- **Module:** `probes/P3-block/block.py`.
  - It needs `probes/P2-script/translit.py` (`to_latin` + `data/work/p2_xlit_table.parquet`) and `probes/P1-taxonomy/normalize.py` (`address_fields().state`).
  - Both are imported by relative probe path in `_latin()` / `_p1()`. Re-point those two imports when promoting to `src/`.
- **Fit (train text only):** `vecs = fit_views(<1M-row sample of train S2+S3>, CFG)`.
  - Hashed features: `HashingVectorizer` + `TfidfTransformer`, 2^18 name / 2^20 addr buckets. A token unseen in train (French street names) gets the maximum IDF instead of vanishing.
  - `full.py fit` does this in 36 s at 1.86 GB and writes `data/work/p3/vecs_hash.pkl`.
- **Run:**
  - Small inputs: `candidates(s1_df, sx_df, CFG, vecs) -> DataFrame[s1, sx, block_score, block_rank_s1, block_rank_sx, via]`.
  - Full scale: `.venv/bin/python probes/P3-block/full.py test <country>`, run once per country.
    - It first builds the texts (P2 → clean → P1 state).
    - It then streams 300k-row chunks through `country_pairs()` and writes `cands_test_<country>.parquet` via arrow.
- **CFG** (the defaults in `block.CFG`):
  - Search: `rev={"na": 10}` is the only sparse product. `k_nostate=30` is the search depth for SX without a state.
  - Keep: `keep_r=2`, `k_noaddr=10`, `keep_f=10`, `views={}` (no real forward).
  - Text and scoring: `w_name=0.5`, `partition=True`, `translit="p2"`, `norm="basic"`, `fold=False`, `hash=True`, `shard=300_000`.
- **`via`:**
  - `r` = the S1 is in the SX's own top-2 (top-10 if the SX has no address).
  - `f` = the SX is in the S1's top-10.
  - `r+f` = both.
  - `block_score` (cosine), `block_rank_s1/sx` and `via` are ready matcher features.
- **For P4/P5:** `data/work/p3/cands_holdout_full_{India,US}.parquet` holds the same rule on the holdout at full-corpus density: 3.1M pairs, with columns s1, sx, block_score, ranks, via, `is_true`. It carries test-like hard negatives, which the dev and holdout mini-worlds lack.
- **Variants:**
  - Cheaper: `keep_r=1, keep_f=5` → 0.9881 at 6.9/S1.
  - Recall-max: `keep_f=20, k_noaddr=30` → 0.9931 at 22.6/S1.

## Evidence
Full-corpus rows use the whole train S2/S3 of the country as the corpus and the whole train S1 of the country as the pool; queries are holdout S1 (fold 0: 88,168 India + 132,190 US, 763,735 true pairs). Test density is 5.8 SX/S1 against train's 4.7, so the test gets ~16/S1.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| full-v4-{India,US}-derived_kr2_kf10_knoaddr10 = holdout-full-export-{India,US} (**recommended**) | holdout-fullcorpus | recall India 0.9906 at 14.0/S1, US 0.9923 at 14.3/S1; S1 full-recall 0.968 / 0.973; S1 with no candidates 0 | keep |
| holdout-full-export-{India,US}-depth10 | holdout-fullcorpus | no-state SX searched only to depth 10: 0.9880 / 0.9904 | drop |
| full-v4-{India,US}-derived_kr1_kf5_knoaddr10 | holdout-fullcorpus | 0.9858 at 6.8 / 0.9896 at 6.9 | cheap variant |
| full-v4-{India,US}-derived_kr2_kf20_knoaddr30 | holdout-fullcorpus | 0.9922 at 22.2 / 0.9936 at 22.8 | recall-max variant |
| full-v4-{India,US}-r:na1 / r:na2 / r:na10 | holdout-fullcorpus | **reverse recall@k** @1/@2/@10: India 0.975/0.982/0.990, US 0.978/0.984/0.992 at 4.7/9.3/46.6 per S1 (@3 0.985/0.987, @5 0.987/0.989) | curve |
| full-{India,US}-recall-curves (v3, w 0.6) | holdout-fullcorpus | **forward recall@k** @5/10/20/50/100: India 0.838/0.925/0.943/0.957/0.965, US 0.822/0.918/0.943/0.962/0.972 | drop (as primary) |
| full-{India,US}-derived_kr2_kf10 (v3, w 0.6, no k_noaddr) | holdout-fullcorpus | 0.9799 at 13.3 / 0.9836 at 13.4 | superseded |
| full-{India,US}-f:na10+r:na2 (v3) | holdout-fullcorpus | real forward + reverse: 0.9816 / 0.9855 at 14.2–14.7. The derived forward is within 0.1–0.2 pt at the same size, for half the compute | drop |
| fulldensity-{India,US}-w{0.4..0.7} | holdout-fullpool | rev@1 India 0.966/0.973/**0.975**/0.972/0.962/0.926 for w 0.4/0.45/0.5/0.55/0.6/0.7; US 0.968 → **0.978** (w 0.6 → 0.5) | keep w=0.5 |
| fulldensity-India-adaptive-k-noaddr | holdout-fullpool | 64% of reverse misses are SX with an empty address. Giving them k=10: 0.9818 → 0.9868 (+1.1 cands/S1) | keep |
| fulldensity-India-extra-views | holdout-fullpool | na@2 0.9818; plus name-only@1 and addr-only@1 gives 0.9822 (2 more products) | drop |
| partition-slot-fraction | train + test sample | the state partition leaves 13% (India), 7.4% (US), 57% (France) of S1×SX slots; state agrees on 99.94% of true pairs after merging tg↔ap and dc↔wa | keep |
| dev-derived-fwd-from-rev10 | dev | per-S1 top-10 taken from the SX→S1 top-10 lists = a real forward top-10 (0.9979 vs 0.9979) | keep |
| dev-loco-{US,India}-fit{other}-{hash,vocab} | dev | IDF fitted on the *other* country (France proxy): US rev@1 vocab 0.9837 → hashed **0.9925** (in-country 0.9934). India 0.9820 → 0.9839 | keep hashed |
| dev-p2+part-* vs dev-basic+part-* | dev | P2 `to_latin` gives cross_script rev@1 0.965 → 0.996 and fwd@20 0.54 → 0.999 | keep |
| dev-basic+part-r:na1 vs dev-basic-r:na1 | dev | state partition 0.9865 → 0.9908: faster **and** better | keep |
| test-cands-{France,India,US} | test | 16.7 / 15.7 / 15.7 per S1; S1 top-1 cosine median 0.90 / 0.97 / 0.93, SX top-1 median 0.79 / 0.82 / 0.77. France looks like US | keep |

**Per P1 class** (recommended rule, full corpus, India+US):

| class | share of pairs | recall | pts of pair recall lost |
|---|---|---|---|
| token_subset | 11.1% | 0.978 | 0.25 |
| no_token_addr_agrees | 1.8% | 0.904 | 0.17 |
| partial_overlap | 5.9% | 0.977 | 0.13 |
| legal_suffix_only | 29.9% | 0.996 | 0.12 |
| exact_norm | 31.5% | 0.998 | 0.06 |
| typo_le2 | 5.2% | 0.993 | 0.04 |
| website | 5.4% | 0.995 | 0.03 |
| cross_script (India only) | 6.5% | 0.997 | 0.02 |
| different_dba / acronym / pipe_joined / reorder | 0.3% combined | 0.52 / 0.96 / 0.98 / 0.999 | 0.03 |

- **Dev overstates recall.** The mini-world overstates full-corpus recall by about 2 points (India 0.9973 dev vs 0.9799 full, same v3 rule). Tune blocking at full density; `revtune.py` does it at ~7% of a full pass.
- **Test run** (`full.py test`, recommended CFG, train-fitted vectorizers; the machine was shared with other probes, load average ~15–25):

  | country | S1 | S2/S3 | pairs | per S1 | search time | peak RAM |
  |---|---|---|---|---|---|---|
  | France | 259,452 | 1,434,993 | 4.33M | 16.7 | 299 s | 1.34 GB |
  | India | 809,986 | 4,717,565 | 12.68M | 15.7 | 1,101 s | 2.68 GB |
  | US | 663,106 | 3,817,031 | 10.42M | 15.7 | 336 s | 2.46 GB |

  Text prep took 388 s for all 11.7M rows (≤ 1.8 GB). **Total ≈ 35 min, measured, not extrapolated.**

## What failed
- **Forward-only S1→SX top-k** (the classic Sparkly direction):
  - Full-corpus recall 0.918–0.925 at k=10 and 0.965–0.972 at k=100.
  - An S2/S3 with an empty address, a website name or an Indic name scores low against every S1, so it drops out of every S1's list. Searched from its own side it still ranks its true S1 first, because the handicap is shared by all its candidates.
  - Each SX has exactly one parent, so the reverse direction is the natural one.
- **No partition:** 403 s per 50k SX against the 1.32M US pool, which extrapolates to ~14 h per country. Killed.
- **max_df pruning** of common n-grams, for speed: 8× less work at max_df 0.02 costs −0.6 pt rev@1. The partition does better on both counts.
- **P1 `name_core` as the name key** (legal forms stripped): −0.3 pt rev@1 on dev, with or without P2. Legal-form n-grams help inside same-name chains.
- **P2 `fold()` on names:** −0.1 pt overall, +0.04 pt cross-script. P2's learned table already gives the S1 spelling.
- **Extra name-only / addr-only views:** +0.04 pt for 2× or 3× the compute.
- **A real forward pass on top of reverse:** +0.1–0.2 pt at equal candidates, for a second full sparse product.
- **Fixed train vocabulary:** drops 32–34% of France's address tokens, which never occur in US/India.
- **Searching no-state SX only to depth 10:** −0.2 pt. Depth is nearly free, and the deep hits still land in an S1's top-10.

## Interfaces
- `block.py`:
  - `fit_views(train_sx, cfg)`
  - `texts(df, cfg) -> [name, addr, st]`
  - `index(ta, vecs, cfg)`
  - `search(ix, tb, vecs, cfg, keep_rev=None)` → hits q1/qx/score/rank/view/dir
  - `country_pairs(ta, chunks, vecs, cfg)` → int pairs, which `finalize()` unions and ranks
  - `candidates(s1_df, sx_df, cfg, vecs)`, plus `fold` and `derive` for evaluation
  - Self-check: `.venv/bin/python probes/P3-block/block.py`.
- `full.py`: `fit` | `prep <country>` | `run <country> <tag>` (holdout evaluation, resumable per chunk) | `test <country>`. Everything intermediate goes to `data/work/p3/` (`txt/`, `txt_test/`, `hits/`, `cands_test_*`, `cands_holdout_full_*`).
- `score_full.py <tag> <country>` scores full-corpus hits on int keys (`classes` builds the P1 taxonomy for the holdout; `export` writes the holdout candidates).
- `revtune.py` is the full-density tuning harness. `partition_cost.py`, the `dev_*.py` scripts and `evalb.py` hold the dev bookkeeping.
- **RAM:**
  - The v4 evaluation steps (run, score, export) stay ≤ 2.3 GB. The v3 run with forward top-100, and its scoring, peaked at 2.7 and 3.2 GB.
  - The test run peaks at 2.7 GB (India). The India S1 index alone is ~1.1 GB, and `CFG["shard"]` is the knob.
- **Time:** 0.09–0.23 ms per SX for the sparse product on 15 threads (US / France / India, growing with the searched S1 slots), plus ~33 µs per row of text prep.

## Risks
- **France (no labels)** — nothing here is validated on France.
  - It has 3 régions only, and 35% of its S2/S3 carry no région or département, so they search all France S1. That is fine for recall but costs 57% of slots.
  - P1's département → région table is domain knowledge.
  - French legal forms (SARL/SAS…) stay in the name n-grams, as legal forms do for US/India.
  - The leave-one-country-out proxy says hashed features lose ≤ 0.1 pt when the IDF comes from another country.
  - On test, France gets 16.7 candidates/S1 with top-1 cosine distributions like the US's.
- **Partition:**
  - An SX whose state disagrees with its S1 is lost; that is 0.06% of true pairs.
  - An S1 without a state joins every group.
  - 1 India test S1 has no candidates; it is output as empty, which is fine for the file format.
- **Licences:** scikit-learn (BSD-3), scipy (BSD), sparse_dot_topn (Apache-2.0), anyascii (ISC), pyarrow (Apache-2.0), numpy/pandas (BSD). P1/P2 modules are stdlib + pandas. No models, no external data, no APIs.
- **IDF** is fitted on train text only. If the organisers allow test-text statistics, refitting on test S2/S3 is a one-line change (`fit_views(test_sx)`) and would help France most.

## Open questions
- May IDF be fitted on test S2/S3 text (PROBLEM.md open question 4)? Best guess: no, so keep hashed train IDF.
- Should candidate_pairs.tsv carry all ~16/S1 or only what the matcher scores? Best guess: the full set. It is the "last set scored", and blocking recall is audited.
- Is the recall-max variant (+0.15 pt at +8 candidates/S1) worth it? Best guess: no. The extra pairs are address-less SX inside chains of same-name S1, and the matcher will abstain on them anyway.
