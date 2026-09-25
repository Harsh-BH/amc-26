# P7-scale — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The full test run fits on the laptop, so no AWS is needed.** One country is resident at a time. The heaviest moment is India, at about 5 GB of a 12 GB budget. The official validator needs 7–8 GB on the full candidate file, so it runs alone afterwards.
- Blocking is the only expensive stage: **~55–65 min** with P3's CFG as-is on a box at load ~20 (France 10.1, US 10.0–13.9, India ~33–41). Everything after blocking takes **~10 min** for ~26M pairs (features 384k pairs/s, LightGBM 100–380k rows/s, writer 51 s for 51M ids).
- Two exact-or-nearly-exact changes to P3's `block.py` cut India search **2.5x at equal load** (765 → 311 s on half SX), putting blocking at **~30 min**. One is splitting S1 pools into ≤ 250k-row pieces (exact up to ties). The other is dropping features whose df in the country's S1 pool exceeds 0.1 (holdout pair recall −0.08 pt). Adopt `chunks.write_lists` as the writer.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **Run shape:** `for country, s1, sx in chunks.by_country(test_source1, [test_source2, test_source3])`, then block → features → predict → decide per country, keeping only `q1, qx` int32 plus prob f32 per pair. Concatenate per-country S1 ids (validator ignores row order) and call `chunks.write_lists` twice. Checkpoint each country's candidates with `chunks.cached(f"data/work/run/{country}_cands.parquet", ...)`. Stop other probes during the run: at load 20 everything ran about 2x slower.
2. **P3 `block.py`, pool pieces (exact):** in `index()`, store every reverse pool as ≤ 250k-row pieces. Pass the list to `topk`, which already accepts pieces:
   `AT[v][g] = [(r[o:o+P], A[v][r[o:o+P]].T.tocsr()) for o in range(0, len(r), P)]`, with `r = ga[g]` (for `""`: `np.arange(len(ta))`) and `P = 250_000`. Then in `search()`, replace `[(ga[g], ix["AT"][v][g])]` with `ix["AT"][v][g]`, which is already the list of (cols, CT) pieces. Measured: India whole-pool calls run 3.0x faster (175 → 59 s), with per-query score lists identical.
3. **P3 `block.py`, own-df prune (optional, if time binds):** `cfg["max_df_own"] = 0.1`. In `index()`, compute
   `keep = (np.bincount(A.indices, minlength=A.shape[1]) <= 0.1 * A.shape[0])` on the country's S1 `na` matrix. Multiply both sides' matrices by it after normalisation (partial cosine). Reference implementation: `probes/P7-scale/bench.py::_prune_mats`. It buys 1.9x (India) to 2.4x (France) at −0.0008 holdout pair recall. P3 should confirm on its full-corpus `full.py` run before adopting.
4. **P3 `candidates()`:** fold the forward pool once at the end, or hold `view`/`dir` as categoricals. The per-shard re-sort costs 141 s of US's 834 s and drives India's RSS.
5. **Text workers:** `chunks.pmap(lambda sl: B.texts(b.iloc[sl], cfg), len(b), 500_000, chunks.workers_for(1.0))`. Build texts once per country, then reuse them in blocking and features. Don't recompute (about 8 min single-thread for the whole test).
6. **Features/predict:** stream 1M-pair chunks with `pmap(step, n_pairs, 1_000_000)` (in-process). Use rapidfuzz `process.cpdist(workers=15)` and numpy compares, never Python per-pair loops. Keep only prob. See SCALE.md "Features → predict → decide".

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| load-parquet | test-full | S2 4.9M rows: pandas 0.65 GB frame / 1.44 GB peak; polars 0.49 / 1.39; astype(object) 1.49 / 2.78 | keep |
| tfidf-fit1M-transform4.9M | test-full | fit 21 s, +0.2 GB; transform 22.6 s/M (linear); CSR 0.24 GB/M rows | keep |
| topn-shard-size | test-India | ns per multiply-add ×threads: 7.3 @125k, 8.0 @250k, 16.7 @500k, 24.5 @1M corpus rows | keep |
| topn-max_df | test-India | 500k corpus, 5k q: 11.4 s → 4.0 s (df 0.05) → 0.8 s (0.01); multiply-adds 1.0e10 → 1.2e9 → 1.6e8 | keep |
| flops-by-feature | test-sample | France: " sa" 23% of name work, "de"+"rue" 55% of addr work; df > 0.1 dropped leaves 6–50% of the work | keep |
| block-full-France | test-France | 10.1 min (search 559 s), 15.03 pairs/S1, 2.5 GB | keep |
| block-full-US | test-US | 13.9 min (search 541, fold 141, index 64, texts 87 s), 14.82 pairs/S1, 4.2 GB | keep |
| block-full-India-oom | test-India | killed at 31 min under the 5 GB cap, unfinished | drop |
| block-India-halfSX | test-India-half | search 953 s → 507 s with the prune; 4.1 GB; full India extrapolated to ~41 / ~25 min | keep |
| block-France-prune0.1 | test-France | search 559 → 236 s (2.4x), 14.79 pairs/S1 | keep |
| topk-pool-pieces | test-India | whole 810k pool: 175 → 59 s (20k q), 55 → 26 s (5k q); same score lists 1.0 | keep |
| dfprune-dev-recall | dev | pair recall 0.9981 → 0.9976 (df 0.1), 0.9975 (0.05); pairs/S1 15.6 unchanged | keep |
| dfprune-holdout-recall | holdout | pair recall 0.9969 → 0.9961, full-S1 0.9893 → 0.9865, 16.5 pairs/S1 | keep |
| pair-features-13 | test-India | 384k pairs/s (6 name + 3 addr rapidfuzz, 2 TF-IDF cosines, 2 length diffs) | keep |
| lgbm-throughput | synthetic | predict 378k rows/s (500 trees, 63 leaves), 99k rows/s (1500, 127); train 1M rows 16–61 s | keep |
| writer+validator | test-full-synthetic | write_lists: 51M ids in 51 s, 2.5 GB, 681 MB; validator PASS; validator ~230 B per id | keep |
| block-full-US-v4 | test-US | 10.0 min (search 448, fold 83 s), 14.67 pairs/S1, 4.4 GB, load 16 | keep |
| block-India-halfSX-v4-prune+pieces | test-India-half | equal load: search 765 → 311 s (2.5x), total 17.1 → 8.4 min, pairs/S1 10.94 → 11.02; full India ~33 → ~16 min | keep |

## What failed      (approaches tried and dropped, with the number that killed them)
- **`multiprocessing.Pool` in `pmap`:** an OOM-killed worker made `map` hang forever (US run stuck for 15 min at 1% CPU). Replaced with `ProcessPoolExecutor`, which raises `BrokenProcessPool`, and the self-check kills a worker to prove it.
- **Forked text workers for India under 5 GB:** OOM (each fork adds ~0.8–1 GB private). At a 12 GB budget, 2–4 workers fit.
- **Full India in one process under 5 GB:** OOM at 31 min. P3's fold pool plus frames exceed 4.3 GB. Measured on half SX instead.
- **Train-df `max_df` prune** (P3's first version): at 0.01 it removed only 37% of France's work, because French stop-tokens ("rue", "de", "sarl") are rare in train. The prune must use the country's own df.
- **polars group_by + str.join writer:** OOM under 5 GB at 51M ids. The existing `s0_exact.write` (dict of sets) would need ~4–5 GB of Python objects.
- **Official validator on a 26M-id candidate file under 5 GB:** OOM, and it needs about 7–8 GB.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
`probes/P7-scale/chunks.py` (self-check: `.venv/bin/python probes/P7-scale/chunks.py`)
- `by_country(s1_path, sx_paths, cols) -> iter[(country, s1_df, sx_df)]`: parquet filter pushdown, countries taken from S1's own labels (unseen labels kept, missing SX gives an empty frame). India resident ≈ 1.3 GB.
- `pmap(fn, n, size, workers=1) -> list`: fn(slice) per row span, in order. With workers > 1 it forks with copy-on-write sharing, and a dead worker raises. With workers=1 it is plain chunked streaming.
- `workers_for(task_gb, reserve_gb=1.0) -> int`: fits workers to min(MemAvailable, cgroup headroom).
- `cached(path, fn) -> DataFrame`: stage checkpoint with atomic write, for resuming a killed run.
- `write_lists(path, col, s1_ids, q1, sx_ids, qx, size=200_000) -> rows`: submission TSV from row numbers. It sorts, de-duplicates, writes every S1 and renames atomically. 51M ids: 51 s, peak +1.5 GB over the id arrays. Passes the official validator.
- `rss_gb()`, `peak_gb()`, `free_gb()`, `countries(path)`.

`probes/P7-scale/bench.py`: one stage per call, one JSON line each: `load`, `tfidf`, `topn`, `flops`, `pieces`, `block <country> [shard workers vecs max_df max_sx piece]`, `recall <max_df> <world>`, `feats`, `lgbm`, `write`. It also holds `_prune_mats` and `_pieces_topk`, monkeypatch reference implementations of recommendations 2 and 3.
Candidates from the benchmark runs (row numbers into the country-filtered test frames) are in `data/work/p7/cands_*.parquet`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Timings come from a shared box** (load 10–25, other probes, this probe capped at 5 GB), so they are pessimistic by about 1.3–2x. India full is extrapolated from half-SX runs (reverse cost is linear in #SX), not measured.
- Assumed that about 12 GB of 15 GB is usable. India blocking and features both need about 5 GB. Running two countries at once gains nothing, because sparse_dot_topn already uses 15 threads.
- P3's CFG moved during the probe (05:10 → v4 at 07:40: w_name 0.5, top-10 for address-less SX). France and US-contended ran on the 05:10 CFG, the holdout recall and the later runs on v4. Cost differences between them are within noise at this load.
- France: 35% of France SX have no parsed state, so they search the whole 260k France pool, which is most of France's cost. A département→region map in P1 `state()` would cut it. The df prune is unvalidated for France (no labels); it drops " sa"/"sarl" 3-grams and "de"/"rue".
- Licences: only already-installed MIT/BSD/Apache libs (pandas, pyarrow, polars, scipy, sparse_dot_topn, rapidfuzz, lightgbm). No external data.

## Open questions   (for the master; each with your best guess)
- Is the df prune worth −0.08 pt pair recall? Best guess: adopt pieces now (free), and add the prune only if the final pipeline's wall time exceeds about 45 min. The F0.5 cost is ~−0.02 pt.
- Should the final run move to SageMaker? Best guess: no. If the laptop must stay free, `ml.m5.12xlarge` (48 vCPU / 192 GB, ~$2.90/h in ap-south-1) runs the whole job in ~20 min for ~$1. Quota exists on the `harsh` account (15 on-demand for most CPU types).
- Holdout evaluation that looks like test (P3 `full.py`) costs ~1.3x a test run. Best guess: run it once per blocking config and cache the candidates. Iterate P4/P5 on the cached candidates.
