# Business entity resolution — Amazon ML Challenge 2026

This code rebuilds both submission files from the raw TSVs. It runs on CPU from the repository root, with no network
access, no external data and no pretrained models. Every intermediate artifact is rebuilt from the training data
when it is absent.

## 1. Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

Unzip `student_resource.zip` so the data sits at `data/student_resource/dataset/{train,test}/*.tsv`, with the official
validator at `data/student_resource/utils/validate_submission.py`. Two environment variables override the paths:
`ER_DATA` (the `dataset/` directory) and `ER_WORK` (intermediate files, default `data/work`).

## 2. Reproduce the submission

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage rules --out output/
```

It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, asserts the format rules, then runs the
official validator. On a cold `data/work` it rebuilds everything below in order; with the caches present it only
matches and writes.

| step | what | module / function | output (in `data/work`) |
|---|---|---|---|
| data | TSV → parquet, read with `QUOTE_NONE` and `keep_default_na=False` | `io_tsv.convert` | `{train,test}_source{1,2,3}.parquet` |
| folds | ground-truth pairs; 10 hashed folds of train S1 (0 = holdout, 1 = dev, 2–9 = fitting) | `worlds.build_base` | `gt_pairs`, `s1_folds` |
| transliteration | Indic → Latin word table, learned from train folds 2–9 true pairs | `translit.table` (+ `pairs.build`) | `p2_xlit_table.parquet` |
| place counts | place frequencies over train S1 addresses (city choice) | `rules.freq` | `place_freq_train.json` |
| record features | `to_latin` → `name_core`, `website_core`, `address_fields` (state, city, house number, places) | `rules.feats` | `rf_test_{s1,sx}_<country>.parquet` |
| vectorizers | hashed TF-IDF (name char-3-grams, address words) on a seeded 1M-row sample of train S2/S3 | `block.vecs` | `p3/vecs_hash.pkl` |
| blocking | per country, within state group: each S2/S3 → its top-2 S1 (top-10 if it has no address) ∪ each S1's top-10 S2/S3 | `block.split_cands` | `p3/cands_test_<country>.parquet` |
| matching | exact (country, name core) or website key + address evidence ≥ 2 → the unique best S1; each S2/S3 to ≤ 1 S1 | `rules.match` | — |
| output | one row per test S1 (all countries); candidates = blocking ∪ accepted pairs | `run.write_lists` | `output/*.tsv` |

Blocking alone can be run per country: `.venv/bin/python code/business_entity_resolution/src/block.py test France`.
Measured on a shared 16-core laptop:
- Blocking: text prep 6.5 min for all 11.7M test rows, then 5 min (France), 18 min (India) and 6 min (US), peaking
  at 2.7 GB. `CFG["shard"]` in `block.py` trades RAM for speed.
- Record features: 20 min for the 12.5M train rows at `PROCS = 2` workers (`run.py`; ~0.5 GB per worker), so about
  the same for the 11.7M test rows.
- Matching and writing, with the caches present: 2.5 min (France), 5 min (India) and 3 min (US), at a 2.3 GB peak.
- The official validator holds every candidate id in RAM, about 5 GB at test scale.

`--stage model` routes the candidates through `model.score(pairs, f1, fx)`, a pair classifier, and then through the
decision layer: `decide.decide` if `src/decide.py` exists, otherwise each S2/S3 goes to its best S1 if p ≥ T.
The classifier and T are fitted on the dev world when `data/work/p8/model_stub.pkl` is absent.

## 3. Validate on the training data

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split holdout --stage rules   # dense holdout F0.5
.venv/bin/python code/business_entity_resolution/src/run.py --split dev --stage model       # small dev world
```

`holdout` is the test-shaped proxy. The fold-0 S1 are scored while every train S1 and every train S2/S3 of the country
compete (2.2M S1, 10.3M S2/S3). The metric is macro F0.5 per S1, singletons included (`metric.py`).

## 4. Tests

```bash
.venv/bin/python code/business_entity_resolution/src/test_pipeline.py      # end to end: every label incl. an unseen one
.venv/bin/python code/business_entity_resolution/src/block.py              # blocking self-check
.venv/bin/python code/business_entity_resolution/src/metric.py             # metric self-check (PDF example)
.venv/bin/python code/business_entity_resolution/src/test_rules_cands.py   # rules guard, dense holdout
.venv/bin/python code/business_entity_resolution/src/package.py --selftest
```

## 5. Package

```bash
.venv/bin/python code/business_entity_resolution/src/package.py --team <team_name>   # -> <team_name>_submission.zip
```

The zip holds `output/`, `code/business_entity_resolution/` (src, this README, requirements) and the filled
`Documentation_template.md`. The script checks its structure and that both TSVs have one row per test S1.
