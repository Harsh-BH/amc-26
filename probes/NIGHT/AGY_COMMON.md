# Common rules for agy probes (read fully before starting)

You are a probe in the Amazon ML Challenge 2026 (business entity resolution, per-S1 macro F0.5; precision counts 2x recall).
Repo root = cwd. Read docs/PROBLEM.md first (rules, metric, data). Python: ALWAYS `.venv/bin/python`, never python3.
pandas 3: string columns are StringDtype (use pd.api.types.is_string_dtype); pandas regex is RE2 (`\w` is ASCII-only,
never strip with [^\w] — it deletes accented letters).

HARD RULES (breaking one disqualifies the team):
- You (the LLM) must NEVER label pairs, judge whether specific pairs match, or produce training data. You write CODE and
  RULES from data STATISTICS only. Reading a few raw rows to understand a pattern is fine; deciding labels is not.
- No external data, no internet data, no hosted APIs in pipeline code. Allowed libs: numpy, pandas, pyarrow, duckdb,
  rapidfuzz, sklearn, lightgbm (already in .venv). Never filter on {US, India}; France is the unseen test country.
- Do NOT edit code/business_entity_resolution/src/. Do NOT git commit / push / checkout. Write only inside your probe
  folder (probes/<ID>/) and data/work/<id-lowercase>/ for intermediates (small files; disk is limited).
- Never write raw data rows (names/addresses) into RESULT.md — aggregate numbers only.

MEMORY: this laptop has ~7 GB RAM shared with other jobs; your process group is capped at 2.2 GB and gets OOM-killed
(exit 137) above it. Load only needed columns, one country at a time, duckdb with `SET memory_limit='1200MB'`. If killed,
shrink and rerun. No GPU.

DATA (all local):
- data/work/{train,test}_source{1,2,3}.parquet: entity_id, business_name, business_address, country (raw).
- data/work/gt_pairs.parquet (s1, sx): train truth. data/work/rf_{train,test}_{s1,sx}_{country}.parquet: normalised
  features (id, core, state, city, hn, ...) from rules.feats (code/business_entity_resolution/src/rules.py, normalize.py).
- data/work/p31/pairs_France.parquet: France test candidate pairs from the P31 France chain with p (matcher prob),
  argmax (SX's best S1), acc (accepted in the OLD #31 base), hn_cls (1 = generator sibling house-number offset,
  0 = equal), name_rel, core_1/core_x, hn_1/hn_x, city_1/city_x, name_1/addr_1/name_x/addr_x (raw), src.
  The current best base (sub46_on43 = sub43_r90 + P46, France 3.256 SX/S1) is NOT on this laptop; it lives on the
  teammate's box. Use P31's acc as a PROXY base for counting, and build a script that applies to ANY base folder.
- data/work/p23/pairs_{US,India}.parquet: holdout pairs of the in-domain model (#8c). USE ONLY rows with holdout==True
  (others have y=False by construction). Columns s1, sx, y (truth), p, am (argmax), pred (accepted).
- data/work/p42/ho_{US,India}.parquet: holdout argmax pairs with structural features + y.
- Prior probes to read (RESULT.md): P46-noise-word-tier (p46.py = template for lens + build), P28-addr-anchor
  (same-address rule), P20-france-patterns (R1 French filler stripping), P40-generator, P41-fr-brand, P42-fr-joint-tune
  (LOCO machinery, solution_p42.py), P37-fr-complete (p37.py LOCO).

VALIDATION BAR (REVISED by the master at 04:10 — the old "precision among pred==False" test is WRONG for France):
In India/US the in-domain model rejects only a tiny, hard remainder of any cell (e.g. 0.1% of it), so precision among
pred==False there says nothing about France, where the out-of-domain model rejects a large share of the cell, mostly
true pairs. Use the NEGATIVES-PER-S1 bound instead:
  neg_per_s1(c) = (# FALSE pairs of the cell on the c holdout, all pairs, accepted or not) / (# holdout S1 of c)
  expected France negatives in the cell  N_neg = neg_per_s1(c) × 259,452 (France S1)
  France add precision  >= 1 − N_neg / (# France pairs the rule would ADD)          (worst case: every negative is added)
Compute it with the US AND the India rate; a rule passes if the US-rate bound >= 0.90 and the India-rate bound >= 0.75
(France addresses behave like the US: 92% of accepted pairs are hn-equal). Also report the cell's overall holdout
precision and its size per S1 in US / India / France (a France cell much LARGER per S1 than US/India means France has
more of something — possibly decoys — and the bound is unreliable: flag it).
Still required: label-free France stats consistent with a true-variant cell, and de-duplication against what
P37/P40/P41/P42/P46 already add (P42 hn-extra = free argmax non-sibling SX, hn equal, p >= 0.80, S1 group non-empty).

DELIVERABLE: probes/<ID>/RESULT.md (TL;DR with numbers, evidence tables, recommendation, what failed) and
probes/<ID>/<id>.py with `build <base_dir> <out_dir>` like p46.py: reads <base_dir>/matching_results.tsv and
candidate_pairs.tsv, adds France pairs only (India/US lines byte-identical), only SX not assigned to any S1 in the base,
only pairs present in candidate_pairs.tsv, caps 5 S2 / 6 S3 per S1, writes out_dir with the same two files. Include a
`selftest` on a tiny synthetic base. Write a draft RESULT.md within your first 30 minutes and keep it updated.
Work autonomously; never ask questions — decide, note the assumption, continue. Finish with RESULT.md final.

MEMORY GOTCHAS (two probes were OOM-killed in their first 3 minutes):
- data/work/p31/pairs_France.parquet with all columns is ~2 GB in pandas. ALWAYS pass columns=[...]; filter to
  argmax rows (argmax==True) with pyarrow filters or duckdb BEFORE pulling the raw name/address strings.
- Raw train sources are 5–6M rows each: never load them whole in pandas; filter by country in duckdb and select 2–3 cols.
- Only ONE python job of yours at a time (no parallel background python). If your python gets killed (137), your agent
  survives: shrink and rerun.
