# P2-script — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- The generator writes Indic names as **token-for-token transliterations of the S1 name** from a closed vocabulary
  (1,518 word types, the same set in train and test). Indic addresses contain nothing but the **state name** (16 phrases).
- A 1,362-entry table learned from train pairs gives the S1 spelling back. Cross-script name AUC goes 0.50 → **0.999**,
  r@5 0.02 → **0.996**, and 94.9% fold to exactly the true S1 name. IndicXlit reaches 0.975 / 0.933, e5-small 0.902 / 0.693.
- Adopt `translit.to_latin` on every S2/S3 name and address **before** normalization, so it feeds both the blocking
  key and the features. Then `fold()` both sides for name n-grams. Add `oov_flag()` as a feature (100% precision distractor flag).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Module:** promote `probes/P2-script/translit.py` together with `probes/P2-script/pairs.py` (DuckDB pair extraction).
  It needs only stdlib + pandas + duckdb, all already in the shared .venv.
- `to_latin(s: pd.Series, tab: dict | None = None) -> pd.Series`
  - Apply it first to every S2/S3 `business_name` and `business_address`, before any normalizer (P1's `name_norm`, `city_key`, etc.).
  - It is a no-op for rows with no Indic text (all of US and France), and it leaves Latin tokens inside mixed rows untouched.
  - Output for Indic spans is lowercase.
  - `tab=None` uses the learned table; `tab={}` gives the rule romanizer only.
- `fold(s: pd.Series) -> pd.Series`: phonetic key (ph→f, w→v, sh/ch→s, z→j, c/q→k, ee/y→i, doubled letters collapsed, accents stripped).
  - Apply it to **both sides** for the name char-n-gram / exact-key comparison.
  - It collapses the table's residual spelling variants (jay/jai, shree/sree, laxmi/lakshmi).
  - Names only: on long addresses it costs ~20 s per 550k rows and adds nothing measurable there.
- `oov_flag(s: pd.Series) -> pd.Series[bool]`: True when an Indic word is outside the table.
  - On dev, 446 records were flagged and **0** of them are matched, so it's a free "distractor" feature for P4/P5.
- **Where it goes: both blocking and features.**
  - Blocking: exact key `country | fold(to_latin(name))`, which hits 94.9% of cross-script pairs. Also add
    `to_latin(name)` into the existing TF-IDF char-gram blocker, where r@5 = 0.996 over the country's S1s.
  - Features: every name feature computed on `to_latin` output; state feature from `to_latin(address)`.
- **Config:** the table is learned on train **folds 2–9**, so dev (1) and holdout (0) stay honest.
  - It is built on first call to `table()` and cached at `data/work/p2_xlit_table.parquet` (+ `p2_pairs.parquet`), both gitignored.
  - Retraining on all folds for the test run is optional: dev OOV was already 0.0% with folds 2–9.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Names.** Dev world: 7,457 cross-script queries (Indic-name S2/S3 with a true S1) vs all 11,994 India S1.
- Similarity: char_wb 2–3-gram TF-IDF cosine, or model cosine for e5.
- AUC is measured against a hard negative: a random S1 that shares a name token with the true S1.
- Ties count against the method.

| ledger name | world | AUC_hard | r@1 | r@5 | r@20 | verdict |
|---|---|---|---|---|---|---|
| name-raw | dev | 0.504 | 0.009 | 0.019 | 0.032 | drop (floor) |
| name-rule | dev | 0.866 | 0.308 | 0.597 | 0.747 | drop |
| name-rule+fold (a) | dev | 0.898 | 0.401 | 0.758 | 0.872 | keep as OOV fallback |
| name-indicxlit (b) | dev | 0.959 | 0.591 | 0.915 | 0.967 | drop |
| name-indicxlit+fold (b) | dev | 0.975 | 0.614 | 0.933 | 0.980 | drop |
| name-dict (c) | dev | 0.998 | 0.746 | 0.992 | 0.9992 | superseded by +fold |
| **name-dict+fold (c)** | dev | **0.999** | **0.752** | **0.996** | **0.9999** | **keep**, exact folded-name hit 0.949 |
| name-e5-small-raw (d) | dev | 0.902 | 0.424 | 0.693 | 0.793 | drop |
| name-latin-ref+fold | dev | 0.958 | 0.856 | 0.943 | 0.956 | reference: noisy Latin S2/S3 of the same S1s |

- **Why r@1 is only 0.75:** 21.8% of the true S1 names are shared by another dev India S1 (common names such as
  "<word> Builders Private Limited"). A name-only comparison cannot separate those; the address has to.
- **Why the table beats even the same-script reference:** the Indic records carry no typos. They are clean
  transliterations of S1, while Latin S2/S3 names are noised.

**Addresses.** Dev world: 9,157 Indic-address queries. Every one has exactly 1 Indic span, and there are only 16 distinct spans, all of them state names.

| ledger name | world | state key = an S1 segment | AUC_hard | r@1 | r@5 | r@20 | verdict |
|---|---|---|---|---|---|---|---|
| addr-raw | dev | 0.000 | 0.994 | 0.923 | 0.962 | 0.981 | keep for whole-address similarity |
| addr-rule | dev | 0.592 | 0.995 | 0.919 | 0.956 | 0.977 | drop |
| addr-dict | dev | **1.000** | 0.995 | 0.915 | 0.954 | 0.975 | keep for the state field |

Brief item 3, the city dictionary, is not needed:
- The city is always Latin in these records; only the state is in Indic.
- Rule romanization misses the exonyms: दिल्ली→"dilli" (Delhi), পশ্চিমবঙ্গ→"pashchimabang" (West Bengal),
  Tamil Nadu, Kerala, Orissa. The 16-phrase state table learned from pairs fixes all of them.
- Whole-address n-gram similarity is unaffected: raw ≈ romanized, since the state is shared by many S1s anyway.

**Other entries.**
- `oov-distractor-flag` (dev): 446 of 12,680 Indic-name records flagged; 0 matched; covers 8.5% of unmatched Indic-name records. Verdict: keep.

**Inventory** (dev world, India, share of rows by dominant script):

| | Latin | Devanagari | Telugu | Tamil | Kannada | Bengali | Gujarati | Malayalam | Gurmukhi | Oriya |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 names | 76.4% | 13.2% | 1.9% | 1.8% | 1.8% | 1.7% | 1.5% | 1.0% | 0.4% | 0.3% |
| S3 names | 86.9% | 7.4% | 1.0% | 1.1% | 1.0% | 0.9% | 0.9% | 0.5% | 0.2% | 0.2% |
| S2 address | 76.7% | 13.6% | 1.6% | 1.7% | 1.9% | 1.6% | 1.4% | 0.8% | 0.3% | 0.3% |
| S3 address | 78.1% | 12.4% | 1.6% | 1.7% | 1.9% | 1.6% | 1.3% | 0.7% | 0.3% | 0.3% |

No other scripts occur. There is no CJK or Arabic anywhere, and France and US have zero Indic characters.

**Structure facts behind the table** (full train, 441k cross-script name pairs from folds 2–9):
- 99.998% of pairs have equal token counts.
- 96.6% positional purity: an Indic token maps to one Latin token. The rest are spelling variants that `fold` merges.
- 1,347 learnable name types. Dev OOV rate: 0.0%.
- Test Indic name vocabulary vs train: 1,518 vs 1,518 types, **0 OOV types**.
- Test Indic address vocabulary: 33 types, of which 1 is OOV (a stray `"`-suffixed token, share 8e-7).

## What failed      (approaches tried and dropped, with the number that killed them)
- **(a) `indic-transliteration`/sanscript as the romanizer.** Dropped before scoring:
  - It needs per-script scheme calls, leaves ZWNJ in the output, maps Tamil த→"dh", and ignores the Bengali nukta.
  - Replaced by the offset-table romanizer in translit.py (all 9 scripts, one table, final-schwa rule for Indo-Aryan scripts).
  - That romanizer scores r@5 0.758, versus 0.996 for the table.
- **(b) IndicXlit (ai4bharat, MIT, ~11M params).**
  - Quality: AUC 0.975 / r@5 0.933. That's better than the rules but 6 points of r@5 below the table, and it still says "dilli" and "poschimbongo".
  - Speed: 18.5 words/s on CPU (8 threads, beam 4, 0.49 GB RSS), plus 53 s to load the model.
  - Uncached, the ~1.2M Indic names (~4.4M words) would take ~66 CPU-hours. Cached per word type (1.5k), it takes ~80 s.
  - It needs Python ≤3.10 + fairseq in a separate venv. Not worth it.
  - GPU throughput was **not measured**. I installed CPU torch on purpose, because CUDA torch (~5 GB) would have filled
    the shared disk (95% full, 9 GB free).
- **(d) multilingual-e5-small (MIT, 118M):** cosine on raw Indic vs Latin: AUC 0.902, r@20 0.793, 1.6k strings/s fp16 on the RTX 3050.
- **(d) LaBSE (Apache-2.0, 471M): not scored.**
  - The first load was OOM-killed: it read fp32 weights into RAM, 1.8 GB, mostly a 501k-token embedding table.
  - The `model.safetensors` download never finished (the session died mid-download), and there's no `accelerate` to load straight to GPU.
  - I did not re-download 1.9 GB onto a 96%-full shared disk, because no embedding can close the gap to the table.
    To run it anyway: download that file, then `flock /tmp/amc-gpu.lock .venv/bin/python probes/P2-script/eval_embed.py sentence-transformers/LaBSE`
    (needs ~2.5 GB free RAM, or `pip install accelerate`).
- **Romanizing the address for whole-address similarity:** r@1 went 0.923 raw → 0.915 table. The state only adds
  similarity to same-state negatives, so use it as a separate state field, not for similarity.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `translit.to_latin(s, tab=None) -> Series[string]`
  - Full test S2+S3 names and addresses: 4 columns, 20M cells, 1.97M of them Indic. 27 s total on one core (~7 s per 5M-row column).
  - 0 Indic characters left afterwards.
  - Peak 1.8 GB, which is mostly the two-column DataFrame. The function itself is O(unique words) thanks to its cache.
  - Measured with `bench.py`.
- `translit.fold(s) -> Series[string]`: 6.2 s for the 547k Indic S2 test names. Per-unique-string LRU cache, Python `re`
  (RE2 can't do backrefs or lookaheads).
- `translit.oov_flag(s, tab=None) -> Series[bool]`: same cost as `to_latin`.
- `translit.learn(pairs) -> dict`: pairs have columns `xn, xa, sn, sa`.
- `translit.table() -> dict`: lazy. Cold build takes 12 s (DuckDB pair extraction 3 s + learn), peaks at 1.09 GB, and is deterministic.
- `translit.roman(token) -> str`: rule romanizer for one Indic token; lru-cached.
- `pairs.build(out)` → `data/work/p2_pairs.parquet`: 1.1M India Indic S2/S3 rows joined to their S1, with the fold. 3 s, under 1 GB (DuckDB memory_limit 900 MB).
- Harnesses:
  - `eval_names.py`: `cross_script`, `hard_negatives`, `evaluate`, `_scores`.
  - `eval_addr.py`, `eval_embed.py` (GPU, under flock), `bench.py`.
  - Metric files: `eval_names.json`, `eval_addr.json`.
- Self-check: `.venv/bin/python probes/P2-script/translit.py`. It asserts the rule romanizer, the learned table on names and addresses, `fold` and `oov_flag`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / external data:** none. translit.py is stdlib + pandas + duckdb (MIT). The table is learned only from the
  provided train files. No model and no download are in the pipeline path. unidecode and aksharamukha are not used.
- **France / unseen countries:** the module is a no-op without Indic codepoints, and it never looks at the country field.
  It won't help an unseen script. For example, Arabic or CJK would pass through untouched; test has neither.
- **The closed vocabulary is a generator artifact.** It's verified on test (0 OOV name types), but if the private
  leaderboard shard came from a different generator, OOV words would fall back to the rules (r@5 0.76 instead of 0.996).
  `oov_flag` would then fire on real matches. Watch its rate on test: ~3.5% of Indic-name rows is normal.
- **The table ships as a derived artifact of Amazon data** in `data/work/`, never in the repo. The final zip has to
  rebuild it from train, which happens automatically on the first `table()` call.
- **Ambiguous tokens** (e.g. Telugu లిమిటెడ్ = limited/ltd, jay/jai) resolve to the majority spelling. The downstream
  legal-form normalizer must treat ltd = limited and pvt = private anyway.
- **Assumption:** holdout honesty needs folds 0 and 1 kept out of the table. Kept out.

## Open questions   (for the master; each with your best guess)
1. **Should the test run's table be rebuilt on all 10 folds?** Best guess: no need. Dev OOV was already 0% and test OOV is 0 types.
2. **Should `oov_flag` be a hard veto or a feature?** Best guess: a feature. It was 0/446 matched on dev, but it's an
   artifact that a different private-shard generator could break. Let P5 decide with its threshold sweep.
3. **Side finding for P4/P5, outside my brief:** some *Latin* S2 name tokens are also distractor-only in train.
   eastgate and northside have 0.0% match rate; southside, westgate, lakeside, riverside and midtown are ≤0.4%; each
   token has ~9.5k occurrences. Base match rate is ~73%. Best guess: a generator distractor vocabulary. P1/P4 should
   confirm out-of-fold before using it.
4. **P3-block logged `recall_cross_script` 0.37–0.98** (ledger 03:50), which is before the learned table existed.
   Best guess: rerunning P3 with the current `to_latin` + `fold` should push its cross-script recall to ≈ its India recall.
