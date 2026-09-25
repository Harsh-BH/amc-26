# Business entity resolution — Amazon ML Challenge 2026

This code rebuilds both submission files from the raw TSVs, from the repository root. It uses no external data and
no API. The one pretrained model is `intfloat/multilingual-e5-small` (MIT, 118M parameters). Its base weights are
downloaded once from Hugging Face, then fine-tuned on the train data. Every other artefact is fitted on the train
data and rebuilt when absent.

## 1. Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt
HF_HUB_OFFLINE=0 .venv/bin/python -c "from transformers import AutoModel, AutoTokenizer as T; \
  AutoModel.from_pretrained('intfloat/multilingual-e5-small'); T.from_pretrained('intfloat/multilingual-e5-small')"
```

After that download, `rerank.py` sets `HF_HUB_OFFLINE=1`, so the pipeline never touches the network.

Unzip `student_resource.zip` so the data sits at `data/student_resource/dataset/{train,test}/*.tsv`, with the official
validator at `data/student_resource/utils/validate_submission.py`. Two environment variables override the paths:
`ER_DATA` (the `dataset/` directory) and `ER_WORK` (intermediate files, default `data/work`).

Hardware: the measurements below are from a shared 16-core laptop, 15 GB RAM, one RTX 3050 6 GB. Every step fits
a 3 GB memory cap. The cross-encoder steps need the GPU; they take the lock `/tmp/amc-gpu.lock` (`flock`).

## 2. Reproduce the submission

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --out output/
```

It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, asserts the format rules, then runs the
official validator. Nothing else has to run first: on a cold `data/work` the command rebuilds every step below in
order (training included). With the caches present it only scores, decides and writes. From the cached X files
(M2 and the cross-encoder recomputed) that takes 35.5 min at a 2.3 GB peak, and reproduces submission #4 byte for
byte.

| step | what | module / function | cached at (`data/work/…`) |
|---|---|---|---|
| data | TSV → parquet, read with `QUOTE_NONE` and `keep_default_na=False` | `io_tsv.convert` | `{train,test}_source{1,2,3}.parquet` |
| folds | ground-truth pairs; 10 hashed folds of train S1 (0 = holdout, 1 = dev / early stopping, 2–9 = fitting) | `worlds.build_base` | `gt_pairs`, `s1_folds` |
| transliteration | Indic → Latin word table, learned from train folds 2–9 true pairs | `translit.table` | `p2_xlit_table.parquet` |
| record features | `to_latin` → `name_core`, `website_core`, `address_fields` (state, city, house number, places) | `run.feats` → `rules.feats` | `rf_<split>_{s1,sx}_<country>.parquet` |
| vectorizers | hashed TF-IDF (name char-3-grams, address words) on a seeded 1M-row sample of train S2/S3 | `block.vecs` | `p3/vecs_hash.pkl` |
| blocking | per country, within state group: each S2/S3 → its top-2 S1 (top-10 if it has no address) ∪ each S1's top-10 S2/S3 | `block.split_cands` | `p3/txt*/`, `p3/cands_<split>_<country>.parquet` |
| rule pairs | exact (country, name core) or website key + address evidence ≥ 2 → the unique best S1 | `rules.match` | — (inside the features) |
| pair features | 65 label-free features per candidate pair (P3 ∪ rule-accepted): name / address similarities, house number / city / state agreement, website, chain size, and each pair's rank and margin among its S1's and its S2/S3's candidates | `features.build` | `p4/X/<split>_<country>_sx_NNN.parquet` |
| M2 | LightGBM on the fold-2 S1 pairs, early-stopped on fold 1 | `train_matcher.main` | `p4/m2_ladder.txt` |
| M2 scores | p for every candidate pair | `model.split_probs` | `p9/m2_<split>_<country>_<booster hash>.parquet` |
| cross-encoder | e5-small fine-tuned on "name \| addr" pairs from M2-like uncertain bands of train folds 4–9 (India, US) | `rerank.py band`, `rerank.py train` | `p4b/trainband_*.parquet`, `p4b/ce.pt` |
| rerank | India / US pairs with 0.02 < p < 0.98: p′ = σ(0.636 logit p + 0.757 logit r + 0.355) | `run.rerank` → `rerank.py score` | `p9/r_<split>_<country>_<ce hash>.parquet` |
| decision | each S2/S3 to its argmax-p S1, kept if p ≥ t. India / US: t = 0.70 on p′. France or any label without training data: M2's p, t from P6's label-free rule (below) | `run.thresholds` | — |
| output | one row per test S1 (all countries); candidates = every scored pair | `run.write_lists` | `output/*.tsv` |

**France threshold, label-free.** France has no labels, and a model applied to an unseen country is over-confident
(leave-one-country-out: US→India loses 4 pts at the source threshold). So the threshold is where France's predicted
empty-S1 rate equals the labelled countries' S1-weighted rate under M2 at M2's own optimum `T_M2` = 0.71. It is
clipped to [0.71, 0.91], and computed from the test probabilities on every run. For the submission it gives 0.82.

Runtimes and memory (measured, 3 GB cap):

| step | cold | peak RAM |
|---|---|---|
| text prep + blocking, test | 6.5 min + 5 / 18 / 6 min (France / India / US) | 2.7 GB |
| record features, test (`PROCS = 2`) | ~20 min | 1.5 GB |
| pair features, test (`features.build`) | 3 / 20 / 9 min (France / India / US) | 2.8 GB |
| TRAIN side for the models: blocking for all 2.2M train S1, record + pair features | ~35 + 20 + 50 min | 2.9 GB |
| M2 training (`train_matcher.py`: 3.14M fold-2 pairs, 793k fold-1 validation pairs) | 18 min | 2.7 GB |
| cross-encoder: `band` ×2 + `train` (GPU) | 5 + 25 min | 2.6 GB, 2.4 GB VRAM |
| M2 scores, test (27.5M pairs from the X files) | 2.3 / 7 / 6 min (France / India / US) | 1.8 GB |
| CE rescoring, test (886k India + 697k US band pairs, GPU) | 11 + 7 min | 2.0 GB (own process), 3.4 GB VRAM |
| decision + writing + validator | 2.5 min | 2.3 GB |

## 3. Validate on the training data

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split holdout --stage model   # dense holdout F0.5
.venv/bin/python code/business_entity_resolution/src/run.py --split dev --stage model       # small dev world
.venv/bin/python code/business_entity_resolution/src/train_matcher.py eval data/work/p4/m2_ladder.txt   # M2 alone
```

`holdout` is the test-shaped proxy. The fold-0 S1 (220,358) are scored while every train S1 and every train S2/S3
of the country compete (2.2M S1, 10.3M S2/S3). Each S2/S3's argmax sees all its train S1: 11.5M pairs are
scored, and 979k band pairs go to the cross-encoder. The metric is macro F0.5 per S1, singletons included
(`metric.py`). The run also refits the blend weights on holdout half 0, as a check.
`dev` runs the same stages in memory (blocking, features, M2, CE) on the 30k-S1 dev world.

| run | F0.5 | time (cold / cached) | peak RAM |
|---|---|---|---|
| `--split holdout --stage model` | **0.9884** (India 0.9894, US 0.9876; half 0 0.9885, half 1 0.9882; P 0.9975, R 0.9697) | 22 min / 1 min | 2.2 GB |
| `--split dev --stage model` | 0.9949 (plain world) | 1.7 min | 0.8 GB |
| `train_matcher.py eval` M2 alone, fold argmax | 0.9813 (`m2_ladder.txt`) = 0.9813 (retrained) | 7.5 min for both | 2.5 GB |
| `--split holdout --stage rules` | 0.7708 | 21 min cold | 2.1 GB |

The blend refit on holdout half 0 returns [0.636, 0.757, 0.355], which is `rerank.CFG["w"]`.

## 4. Tests

```bash
.venv/bin/python code/business_entity_resolution/src/test_pipeline.py      # end to end: every label incl. an unseen one
.venv/bin/python code/business_entity_resolution/src/features.py           # feature self-check
.venv/bin/python code/business_entity_resolution/src/rerank.py check       # blend-rule self-check
.venv/bin/python code/business_entity_resolution/src/block.py              # blocking self-check
.venv/bin/python code/business_entity_resolution/src/metric.py             # metric self-check (PDF example)
.venv/bin/python code/business_entity_resolution/src/package.py --selftest
```

`--stage rules` reproduces submission #2, the rules-only baseline, byte for byte.

## 5. Package

```bash
.venv/bin/python code/business_entity_resolution/src/package.py --team <team_name>   # -> <team_name>_submission.zip
```

The zip holds `output/`, `code/business_entity_resolution/` (src, this README, requirements) and the filled
`Documentation_template.md`. The script checks its structure and that both TSVs have one row per test S1.
The fine-tuned weights (`data/work/p4b/ce.pt`, 43 MB) and the booster (`data/work/p4/m2_ladder.txt`, 32 MB) are
artefacts, not source. Both are rebuilt from the train data by the commands above.
