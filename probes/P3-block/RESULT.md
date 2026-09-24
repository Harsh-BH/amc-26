# P3-block — RESULT
## TL;DR
DRAFT 2 (05:00). Dev-world numbers are final; the full-corpus holdout run (10.3M S2/S3, 2.2M S1 pool) is being re-run under the 2.3 GB step cap.
The best blocker is one sparse product per country: for each S2/S3, its top-10 S1 by TF-IDF cosine (name char-3grams + address words). Name and address are transliterated with P2 first, and the search is restricted to the SX's state group.
Dev: reverse top-1 gives pair recall 0.9929 at 5.8/S1. Adding reverse top-2 and a per-S1 top-10 taken from the same lists gives 0.9979 at 15.6/S1 (cross-script 0.999).

## Recommendation
Provisional: `probes/P3-block/block.py::candidates(s1_df, sx_df, cfg, vecs)` with
`cfg = {**CFG, "views": {"na": 10}, "rev": {"na": 10}, "partition": True, "translit": "p2"}`, then keep pairs with
`block_rank_sx < 2` or (forward rank < 10). A per-S1 top-10 derived from the reverse lists equals a real forward pass: 0.9979 vs 0.9979 on dev. The full-corpus number will confirm it.

## Evidence
All rows: vectorizers fitted on train S2/S3 only, within country. `f:X k` = S1→SX top-k; `r:X k` = SX→S1 top-k.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| dev-f:na20 (no translit, no partition) | dev | recall 0.931 at 20/S1; cross_script 0.372 | drop |
| dev-r:na1 | dev | recall 0.987 at 5.8/S1; cross_script 0.964 | keep (direction) |
| dev-basic+part-r:na1 | dev | recall 0.9908 at 5.8/S1 (vs 0.9865 without the partition) | keep |
| dev-p2+part-r:na1 | dev | recall 0.9929 at 5.8/S1; cross_script 0.9960 | keep |
| dev-p2+part-f:na10+r:na2 | dev | recall 0.9979 at 16.6/S1; full-recall 0.9929; cross_script 0.9991 | keep |
| dev-p2+part-f:na10,name5,addr5+r:na2,name1,addr1 | dev | recall 0.9986 at 20.7/S1; no_token_overlap 0.979 | keep (costs 3 products) |
| dev-p1+part-r:na1 | dev | recall 0.9876 (P1 name_core as the name key) | drop |
| dev-p1+p2+part-r:na1 | dev | recall 0.9903 vs 0.9929 for p2 with basic cleaning | drop |
| dev-prune-maxdf-sweep | dev | r:na1 0.9867 → 0.9804 at max_df 0.02 (8× less work) | drop |
| dev-derived-fwd-from-rev10 | dev | derived forward top-10 + r2: 0.9979 at 15.6/S1 = real forward | keep |

## What failed
- **Forward-only S1→SX top-k**: 0.931 at k=20, 0.957 at k=100 (dev). An SX with an empty address or an Indic name scores low against every S1, so it falls out of the S1's top-k. Ranking per SX cancels that handicap, so the reverse direction wins.
- **max_df pruning** of high-frequency n-grams (for speed): each halving of work costs about 0.3–1 point of reverse recall. Dropped in favour of the state partition, which is faster and more accurate.
- **P1 name_core** (legal forms and honorifics stripped) as the name key: 0.25–0.3 points worse than plain cleaning at r:na1, with or without P2. Legal-form n-grams help separate look-alikes inside a state.
- **Full-corpus run without the partition**: 50k SX queries against the 1.32M US S1 pool took 403 s, which extrapolates to ~14 h per country. Killed.

## Interfaces
- `block.py`: `fit_views(train_sx, cfg) -> {name, addr: TfidfVectorizer}`, `texts(df, cfg)`, `mats(...)`, `topk(Q, CTs, k, cfg)`,
  `run_views(a, b, vecs, cfg, fwd_rows, keep_rev)` (one country, SX streamed in shards), and `candidates(s1_df, sx_df, cfg, vecs)`.
- `evalb.py` (recall bookkeeping), `full.py` (full-corpus holdout run), `score_full.py` (combo grid).
- Full-scale RAM and time: pending (re-running per country in chunks).

## Risks
- Partition: an SX whose state group differs from its S1's is lost. That is 0.06% of true pairs after merging tg↔ap and dc↔wa. An SX with no state (4%) searches the whole country pool.
- France (no labels): the state groups come from P1's région/département table. IDF is fitted on US/India, so French address words never seen in train are dropped from the address view (under investigation).
- Licences: sparse_dot_topn (Apache-2.0), scikit-learn (BSD), anyascii (ISC); P2 translit is stdlib only.

## Open questions
- May IDF be fitted on test S2/S3 text (PROBLEM.md open question 4)? Best guess: no. Use hashed features, where unseen tokens get the maximum IDF.
