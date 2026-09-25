# SCALE — the full test run on the 15 GB laptop

Target: test = 1,732,544 S1 + 9,969,589 S2/S3 (India 0.81M / 4.72M, US 0.66M / 3.82M, France 0.26M / 1.43M).
Box: 16 cores, 15 GB RAM. Assume about 12 GB is usable for the run (desktop plus one claude CLI take the rest).
**All timings come from a shared box**: load average 10–23 from other probes, and this probe was capped at
5 GB. They are pessimistic by roughly 1.3–2x against a quiet machine. Raw rows: `ledger.py show exp`, probe P7-scale.
Blocker measured = P3 `block.py` CFG as of 05:10: reverse SX→S1 `na` top-10, derive keep_r=2 / keep_f=10,
state partition, P2 translit, hashed TF-IDF.

## Verdict

**It fits locally, and AWS is not needed.** One country is resident at a time, and the largest (India) peaks
around 5–6 GB with P3's code (it went past this probe's 5 GB cap). Blocking takes **~1 h** end to end on a busy box, and
**~30 min with the df prune plus pool pieces** below. Everything after blocking takes **< 10 min** for ~26M pairs.
The run must not share the box with probes: at load 20 every stage here ran about 2x slower.

## Stage budget (per-country loop)

| # | stage | measured | full test | peak RAM | knob |
|---|---|---|---|---|---|
| 0 | load parquet with `filters=[("country","==",c)]` | S2 4.9M rows: 0.65 GB frame, 1.44 GB peak (pandas pyarrow str); polars 0.49 / 1.39 | < 5 s | India ~1.3 GB | never `astype(object)`: 1.49 GB frame, 2.78 GB peak |
| 1 | texts (P2 translit + P1 state + clean) | S1 22 s/M, SX ~44 s/M, single thread | ~8 min, 1 thread | +0.8–1 GB per forked worker | `pmap` workers: 4 is fine for France/US at 12 GB; India forks OOM'd at 5 GB |
| 2 | TF-IDF transform (inside P3 `search`) | 22.6 s/M, linear (1M → 4.9M); CSR 0.24 GB per M rows (name 21.6 + addr 7.6 nnz/row) | ~4.5 min (part of search time) | one shard | fit on 1M train rows: 21 s, +0.2 GB |
| 3 | **blocking** | France 10.1 min, US 10.0–13.9 min, India ~33–41 min (half-SX ×2) | **~55–65 min** (~30 with prune + pieces) | France 2.5, US 4.4, India ~5 GB | shard 500k, pieces ≤ 250k, see below |
| 4 | candidate store | 14.8–15.6 pairs/S1 (France 15.0, US 14.8, dev 15.6) → **~26M pairs** | — | int32 q1/qx + f32 score: 0.3 GB | keep row numbers, never id strings, until the writer |
| 5 | pair features (13: 6 name fuzz, 3 addr fuzz, 2 TF-IDF row cosines, 2 length diffs) | **384k pairs/s** (rapidfuzz `process.cpdist`, workers=15) | 26M: 70 s; 50M: 130 s | 1M-pair chunk ~0.3 GB | no Python per-pair loops: parse per record, compare vectorised |
| 6 | LightGBM predict | 378k rows/s (500 trees, 63 leaves, 30 f); 99k rows/s (1500 / 127 / 50 f) | 26M: 1–4.5 min | < 0.7 GB | 1M-row chunks; train 1M rows: 16–61 s |
| 7 | write both TSVs | see Writer | | | `chunks.write_lists` |

## Blocking

### Where the time goes
The reverse top-k costs `Σ_f df_Q(f)·df_C(f)` multiply-adds, and each one costs more as the corpus per call grows
(India, 5k queries, 15 threads):

| corpus rows per sp_matmul_topn call | 125k | 250k | 500k | 1M |
|---|---|---|---|---|
| ns per multiply-add × threads | 7.3 | 8.0 | 16.7 | 24.5 |

The state partition keeps most calls small (US 45 groups, India 29, France 4) and cuts US work 10x
(2.0e12 → 2.1e11 multiply-adds per direction). An SX with no parsed state searches the whole country pool.
In France that is **35% of SX** (500k rows against all 260k S1), and it is where France's time goes.
A France département→region map in P1's `state()` would move most of them into a group.

### A handful of features carry most of the work
From 100k S1 × 200k S2 per country, df = max(df_S1, df_S2) within the country:

| country | biggest features (share of multiply-adds) | work left after dropping df > 10% (name / addr) |
|---|---|---|
| France | name " sa" 23%, "rl " 7%; addr "de" 32%, "rue" 23% | 30% / 6% (25 + 18 features) |
| India | name "limited" / "private" 3-grams ~7% each; addr "no" 31% | 8% / 29% (28 + 12) |
| US | name "llc" / "inc" 3-grams ~7% each; addr "street" 16%, "road" 13% | 50% / 44% (13 + 8) |

These are legal forms and street words, and they carry no identity. Train-fitted IDF misses them for France (a
train-df max_df of 0.01 removed only 37% of France's work), because French tokens never occur in train.
The prune must use the **country's own df**.

**Prune = drop features whose df in the country's S1 pool exceeds 0.1** (partial cosine: norms stay those of the
full vector; implemented as `bench._prune_mats`, a wrapper around `B.mats`):

| | dev pair recall | dev full-S1 | holdout pair recall | holdout full-S1 | pairs/S1 (holdout) | France search s (full test) |
|---|---|---|---|---|---|---|
| no prune | 0.9981 | 0.9935 | 0.9969 | 0.9893 | 16.49 | 559 |
| df > 0.2 dropped | 0.9979 | 0.9925 | — | — | — | — |
| **df > 0.1 dropped** | **0.9976** | **0.9917** | **0.9961** | **0.9865** | 16.47 | **236 (2.4x)** |
| df > 0.05 dropped | 0.9975 | 0.9913 | — | — | — | — |

Dev = P3 CFG at 05:10 (w_name 0.6). Holdout (220k S1 at test density) = P3 CFG v4 at 07:40 (w_name 0.5, top-10 for
address-less SX). Cost of the prune: −0.08 pt pair recall, −0.28 pt of S1 with every true pair found. Worth it
only if blocking time binds: it buys 1.9x (India) to 2.4x (France). P3 should confirm on its full-corpus `full.py`
run, because dev and holdout-world recall run about 2 pt above the full corpus.

### Per-country blocking time (full test; load 13–23)

| country | S1 × SX | texts | index | search | fold | total | pairs/S1 | peak |
|---|---|---|---|---|---|---|---|---|
| France | 0.26M × 1.43M | 21 s | 4 s | 559 s | 22 s | 10.1 min | 15.03 | 2.5 GB |
| France, prune 0.1 | | 62 s | 10 s | 236 s | 15 s | 5.4 min | 14.79 | 2.3 GB |
| US | 0.66M × 3.82M | 87 s | 64 s | 541 s | 141 s | 13.9 min | 14.82 | 4.2 GB* |
| India, half SX (2.36M) | 0.81M × 2.36M | 205 s | 37 s | 953 s | 99 s | 21.6 min | 10.94 | 4.1 GB |
| India, half SX, prune 0.1 | | 193 s | 26 s | 507 s | 75 s | 13.4 min | 11.02 | 3.9 GB |
| India full, prune 0.1 (extrapolated) | | 336 s | 26 s | 1015 s | 150 s | ~25 min | | ~4.8 GB |
| *same load (~20), P3 CFG v4:* | | | | | | | | |
| US, v4 | 0.66M × 3.82M | 58 s | 13 s | 448 s | 83 s | 10.0 min | 14.67 | 4.4 GB* |
| India half SX, v4 | 0.81M × 2.36M | 149 s | 52 s | 765 s | 57 s | 17.1 min | 10.94 | 4.0 GB |
| India half SX, v4, **prune 0.1 + pieces ≤ 250k** | | 119 s | 19 s | **311 s (2.5x)** | 52 s | **8.4 min** | 11.02 | 4.2 GB |
| **India full, v4 (extrapolated: search, SX texts and fold ×2)** | 0.81M × 4.72M | 262 s | 52 s | 1531 s | 114 s | **~33 min** | ~15 | ~5 GB |
| **India full, v4, prune + pieces (extrapolated)** | | 213 s | 19 s | 622 s | 104 s | **~16 min** | | ~5 GB |

\* the benchmark held all shard texts at once (about +1 GB against P3's lazy loop). Full India single-process was
OOM-killed at 31 min, still unfinished, under this probe's 5 GB cap. Hence the half-SX runs: reverse-search cost is
linear in #SX because the S1 pool is fixed. The S1 texts, the index and the S1×keep_f pool are the same at half and full.

**Split big S1 pools into ≤ 250k-row pieces** before `topk` (P3's `topk` already takes a list of pieces). Every
no-state SX searches the whole country pool, and India's big states hold > 100k S1 each. Measured on the whole
India pool (810k S1, P3 `B.topk`, k=10): 20k queries 175 s → **59 s** (3.0x); 5k queries 55 s → 26 s (2.1x).
Per-query score lists are identical (1.0). Only tied members swap (1.5% of pairs), so the result is exact up to ties.

P3's `fold` re-sorts the whole running pool after every shard: 141 s for US, ~17% of its time. Doing it once at the
end, or holding `view` / `dir` as categoricals instead of object strings, cuts both the time and India's RSS.

## Features → predict → decide, streamed (design)

Per country, resident: texts + parsed per-record fields for S1 and SX (India ~1.5 GB), the name/addr CSR for
S1 and SX (India 5.5M rows × 0.24 GB/M = 1.3 GB), and the candidates as `q1, qx` int32 plus `block_score` f32.
Then:

```python
o = np.argsort(q1, kind="stable")                       # pairs sorted by S1 row
def step(sl):                                            # 1M pairs: ~0.3 GB strings + 0.16 GB features
    i, j = q1[o][sl], qx[o][sl]
    F = features(i, j)                                   # rapidfuzz cpdist(workers=15) + numpy compares
    return model.predict(F, num_threads=15).astype(np.float32)
prob = np.concatenate(pmap(step, len(o), 1_000_000))     # workers=1: in-process streaming, results in order
```

Features never exist for all pairs at once: 50M × 40 f32 would be 8 GB. Only `prob` (4 B/pair) survives.
Training uses the same `features()` on the train world (200k S1 × ~15 = 3M pairs, 0.5 GB, fits in RAM).
Decision + 1-to-many exclusivity: one `lexsort` by (qx, -prob) on ≤ 26M rows takes seconds.

## Writer and validator

`chunks.write_lists(path, col, s1_ids, q1, sx_ids, qx)` works from row numbers. It packs (q1, qx) into one int64 key,
sorts in place, de-duplicates and writes 200k S1 per chunk to a temp file that is then renamed. Every S1 gets a row.

| | measured |
|---|---|
| 1.73M S1, 51M candidate ids (synthetic, 0–60 per S1) | 51 s, 681 MB file, peak 2.5 GB (1.5 GB above the id arrays) |
| both files, 26M candidates + 1.4M matches | 22 s, peak 2.0 GB |
| polars group_by + str.join, 51M ids | OOM-killed under 5 GB |
| existing `s0_exact.write` (dict of Python sets) | ~4–5 GB of str + set objects at 26M ids (not run) |
| **official validator on write_lists output (6.9M cand ids)** | **PASS**, with and without `--check-ids` |
| validator peak RAM | 2.29 GB for 6.9M + 1.6M ids (~230 B per id); 3.18 GB with `--check-ids` |
| validator on 26M candidate ids | OOM-killed under 5 GB; **needs ~7–8 GB: run it alone, after the pipeline exits** |

## Total wall time (full test, this box, load ~20)

| config | France | US | India | blocking | after blocking | total |
|---|---|---|---|---|---|---|
| P3 CFG as it is | 10.1 (measured) | 10.0–13.9 (measured) | ~33–41 (half-SX ×2) | **~55–65 min** | ~10 min | **~65–75 min** |
| + df prune 0.1 + pool pieces ≤ 250k | 5.4 (measured, prune only) | ~8 (est.) | ~16 (half-SX ×2) | **~30 min** | ~10 min | **~40 min** |
| same on a quiet box (no other probes; est. 0.5–0.7x) | | | | ~15–20 min | ~6 min | **~25 min** |

"After blocking" for ~26M pairs: texts and parsed fields cached from blocking (else +8 min single-thread), P4
features at ~3x the 13 benchmarked (~4 min), LightGBM 1–4.5 min, decision + exclusivity < 1 min, both TSVs ~25 s.

**RAM at 12 GB:** the heaviest moments are India blocking (~5 GB) and India features (frames 1.3 + texts 1.5 +
CSR 1.3 + candidates 0.2 + one 1M-pair chunk 0.5 ≈ 5 GB). Both leave 7 GB of headroom, so one country at a time
is safe. Running two countries at once does not help: sparse_dot_topn already uses 15 threads. The validator
comes after, alone (~7–8 GB).

**Holdout evaluation that looks like test** (P3 `full.py`: all 10.3M train SX against the full 2.2M train S1 pool)
costs ~1.3x a test run: US is 1.32M S1 × 6.2M SX in train. Run it once per blocking config, cache the candidates
(`chunks.cached`), and iterate the matcher on the cached candidates.

## AWS (only if the laptop must stay free, or the blocker doubles)

RAM is not the constraint, cores are: sparse_dot_topn, rapidfuzz and LightGBM all scale with threads. SageMaker
training-job prices in ap-south-1, derived linearly from docs/SAGEMAKER.md's ml.m5.xlarge $0.242/h (m5 pricing is
linear in size):

| instance | vCPU / RAM | $/h | est. full test run | cost per run |
|---|---|---|---|---|
| ml.m5.4xlarge | 16 / 64 GB | ~0.97 | ~same as laptop quiet (~40 min) + 5 min start | ~$0.75 |
| ml.m5.12xlarge | 48 / 192 GB | ~2.90 | ~15 min + 5 min start (3x threads; all 3 countries in parallel fit in RAM) | ~$1.0 |

Move stages 1–3 (texts + blocking), or simply the whole `run test` job; the rest runs in minutes anywhere.
Quota on the `harsh` account: most CPU types have 15 on-demand (memory note, 2026-09-24). Credits: $125 left there;
a full run costs ~$1. The limit is setup time (image + pip of sparse_dot_topn / rapidfuzz / lightgbm
at job start), not money.
