# Setup — run the pipeline on your own machine

For teammates (team **Yeti**). One command rebuilds both submission files from the raw data. Start with
[APPROACH.md](APPROACH.md) if you want to know what the pipeline does; this page is only how to get it running.

## 0. Quick start — pick your path

| Path | You download | Run time (16-core laptop, RTX 3050) |
|---|---|---|
| **A. Fastest** | code + data + trained models + cached test features | **a few minutes** |
| **B. Normal** | code + data + trained models | **~2.5–3 h** |
| **C. From scratch** | code + data | **~5.5–6 h** (retrains every model) |

All three give the same output. Steps: **1** access → **2** code → **3** install → **4** data → **5** models (A, B) →
**6** cached features (A) → **7** run.

## 1. Getting access (Harsh does this once per teammate)

Both are private.

| What | How Harsh shares it | What you get |
|---|---|---|
| Code | GitHub → `Harsh-BH/amc-26` → Settings → Collaborators → add your GitHub username | the repo |
| Files | Google Drive → folder `amc26-offload` → right-click → Share → your email (Viewer) | data, models, caches |

Never make the Drive folder "anyone with the link" — it holds a copy of Amazon's dataset.

**What is in the Drive folder `amc26-offload/`:**

```
amc26-offload/
├── data/student_resource/dataset/   the challenge data (train + test TSVs)          ~5 GB
├── models/                          our trained models (step 5)                    ~110 MB
│   ├── p4/m2_big.txt                  LightGBM matcher (the "main judge")
│   ├── p4b/ce.pt                      fine-tuned e5-small cross-encoder (the "second opinion")
│   └── p10/pruner.txt                 candidate pruner (shrinks shortlists 15.9 → 4.8 per business)
└── work/                            cached test-side intermediate files (step 6)    ~4.6 GB
```

Download either from the Drive website (folder → Download) or with rclone:
`rclone config` → new remote named `gdrive`, type `drive`, then `rclone copy gdrive:amc26-offload/<folder> <dest>`.

## 2. Get the code

```bash
git clone https://github.com/Harsh-BH/amc-26.git
cd amc-26
```

## 3. Install (~5–10 min)

| Requirement | |
|---|---|
| OS | Linux or macOS (tested on Arch Linux) |
| Python | **3.12** |
| RAM | 16 GB (every step fits a 3 GB cap, but the OS needs headroom) |
| Disk | ~20 GB free |
| GPU | NVIDIA with ≥ 4 GB VRAM strongly recommended (cross-encoder). Without one it runs on CPU, much slower |

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
# the one pretrained model (MIT, 118M) — downloaded once; the pipeline then runs offline
.venv/bin/python -c "from transformers import AutoModel, AutoTokenizer as T; \
  AutoModel.from_pretrained('intfloat/multilingual-e5-small'); T.from_pretrained('intfloat/multilingual-e5-small')"
```

Always use `.venv/bin/python`, never a system `python3` (pandas 3 + Python 3.12 are pinned; other versions silently
break text handling).

## 4. Get the data

Download `student_resource.zip` from the Unstop challenge page, or copy `amc26-offload/data/student_resource/` from
Drive. It must end up exactly here:

```
data/student_resource/dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/student_resource/dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
data/student_resource/utils/validate_submission.py
```

`data/` is gitignored — **never commit it**, never upload it anywhere public (it is Amazon's dataset).

## 5. Trained models — paths A and B (skips ~3 h of training)

Download `amc26-offload/models/` and place the files exactly here:

| Drive file | Put it at | Size | sha256 |
|---|---|---|---|
| `models/p4/m2_big.txt` | `data/work/p4/m2_big.txt` | 67 MB | `598e51c859cf85bf7d59f75135959fcc47e4c71e05ebe6ae3a2414695e485f03` |
| `models/p4b/ce.pt` | `data/work/p4b/ce.pt` | 42 MB | `16013613bbef1c9548a7d91b73978e0be6f833a6007ab45f2460644e1b55c789` |
| `models/p10/pruner.txt` | `data/work/p10/pruner.txt` | 1 MB | `02ab27da2fca67220fc8377eb881c858aa4871c0eaeacd9efa75bed4eb885a52` |

```bash
rclone copy gdrive:amc26-offload/models data/work/     # or copy the three files by hand into those paths
sha256sum data/work/p4/m2_big.txt data/work/p4b/ce.pt data/work/p10/pruner.txt   # must match the table
```

## 6. Cached test features — path A only (the run takes minutes)

`amc26-offload/work/` mirrors `data/work/` with every **test-side** intermediate file. Copy it in, keeping the
sub-folders:

```bash
rclone copy gdrive:amc26-offload/work data/work/     # or download the folder and merge it into data/work/
```

What it contains, and what each file saves you:

| Cached file (`data/work/…`) | What it is | Time saved |
|---|---|---|
| `test_source*.parquet` | the test TSVs in a fast format | minutes |
| `rf_test_*_<country>.parquet` | record features: cleaned name, city, house number, state | ~20 min |
| `p2_xlit_table.parquet`, `place_freq_train.json`, `p3/vecs_hash.pkl` | transliteration table, place counts, TF-IDF vectorizers (all fitted on train) | minutes |
| `p3/cands_test_<country>.parquet`, `p3/txt*/` | candidate shortlists from blocking | ~36 min |
| `p4/X/test_<country>_sx_NNN.parquet` | the 65 match signals for every candidate pair | ~30 min |
| `p9/m2_test_<country>_<model-id>.parquet` | LightGBM scores | ~1 h |
| `p9/r_test_<country>_<model-id>.parquet` | cross-encoder scores | ~18 min GPU |

**Version rule — important.** The caches were built by commit `cef4e2d` (25 Sep 22:43) or later. Feature caches are
matched **by file name, not by code version**: if your code cleans text differently from the code that built them,
the pipeline silently reuses stale features. Check out the same commit Harsh used when he uploaded the caches
(`git checkout <commit>`), or delete the affected files so they rebuild. Score caches are safe: their file names
contain the model's id, so a new model always re-scores.

## 7. Run

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --out output/
```

It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (1,732,544 rows each — one per test Source 1
entity, France included), checks the format, then runs the official validator. Expect `PASS`.

It rebuilds whatever is missing, in order, and caches it under `data/work/`, so a second run only re-scores and
re-writes. To force a step to rebuild, delete its cache file(s) from the table in step 6.

## 8. How long it takes (measured on a 16-core laptop, RTX 3050)

| Stage | Path A | Path B | Path C |
|---|---|---|---|
| text prep + candidate search (test) | cached | ~36 min | ~36 min |
| record + pair features (test) | cached | ~50 min | ~50 min |
| train-side candidates + features | — | — | ~1 h 45 min |
| train LightGBM matcher | — | — | ~45 min |
| train cross-encoder (GPU) | — | — | ~30 min |
| score test pairs (LightGBM) | cached | ~1 h (less with the pruner) | ~1 h |
| cross-encoder rescoring (GPU) | cached | ~18 min | ~18 min |
| decide + write + validate | ~3 min | ~3 min | ~3 min |
| **Total** | **a few minutes** | **~2.5–3 h** | **~5.5–6 h** |

## 9. Other useful commands

| What | Command |
|---|---|
| Score our own holdout (the number we trust; #5 = 0.9888) | `.venv/bin/python code/business_entity_resolution/src/run.py --split holdout --stage model` |
| Reproduce submission #4 instead | add `--recipe sub4` |
| Pipeline regression test | `.venv/bin/python code/business_entity_resolution/src/test_pipeline.py` |
| Validate any submission (from `data/student_resource/`) | `python3 utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test` |
| Experiment ledger (every run and its score) | `.venv/bin/python ledger.py show exp` · `show pipe` |

## 10. Troubleshooting

- **Text columns vanish / accents or Hindi letters get stripped** — wrong pandas/Python. Use the venv.
- **Out of memory** — close heavy programs; every step is designed for ≤ 3 GB. Linux swap helps.
- **No GPU** — it still runs; the cross-encoder steps are slow on CPU. `rerank.py` takes `/tmp/amc-gpu.lock`, so two
  runs never share the GPU.
- **Hugging Face download fails** — run step 3's download once with internet; after that the pipeline is offline
  (`HF_HUB_OFFLINE=1`).
- **Output differs from Harsh's** — check you are on the same commit and that your caches (step 6) came from it.
- **Drive says "access denied"** — Harsh has not shared `amc26-offload` with your email yet (step 1).

## 11. Rules we must not break

- No external data, no APIs (no geocoding, registries, hosted LLMs), no internet lookups in the pipeline.
- Every model must be MIT / Apache-2.0 and ≤ 8B parameters, run locally.
- Never commit `data/`, `output/` or any `*.tsv`. Details: [docs/PROBLEM.md](docs/PROBLEM.md).
- Leaderboard: 5 uploads per team per day (they reset at midnight IST). Coordinate with Harsh before uploading —
  the queue and every score are in [SUBMISSIONS.md](SUBMISSIONS.md).

## 12. Where things are documented

| File | What |
|---|---|
| [APPROACH.md](APPROACH.md) | the whole approach in plain language |
| [docs/PROBLEM.md](docs/PROBLEM.md) | rules, output format, metric, data facts |
| [SUBMISSIONS.md](SUBMISSIONS.md) | every staged/uploaded submission and its score |
| [Documentation_template.md](Documentation_template.md) | the methodology document for the final package |
| `code/business_entity_resolution/README.md` | pipeline internals: every stage, module and cache |

## 13. Current status (update as it changes)

- The command reproduces **submission #5** byte for byte (holdout 0.9888, leaderboard 0.980017).
- Best leaderboard so far: **0.981024** (France threshold raised to 0.88).
- Being added now: the **candidate pruner** (#6: 4.83 candidates per business instead of 15.9) and a **France
  threshold setting**. Pull the repo again once they are merged.
