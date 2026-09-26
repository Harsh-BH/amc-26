# Setup — run the pipeline on your own machine

For teammates (team **Yeti**). One command rebuilds both submission files from the raw data. Start with
[APPROACH.md](APPROACH.md) if you want to know what the pipeline does; this page is only how to get it running.

## 0. Quick start — pick your path

| Path | You download | Run time (16-core laptop, RTX 3050) |
|---|---|---|
| **A. Fastest** | code + data + trained models + cached test features | **a few minutes** |
| **B. Normal** | code + data + trained models | **~2–2.5 h** |
| **C. From scratch** | code + data + the bge checkpoint | **~5–6 h** (retrains every other model; +~2 h of laptop GPU if the bge scores are not cached) |

All three give the same output. Steps: **1** access → **2** code → **3** install → **4** data → **5** models (A, B; C needs the bge checkpoint from it) →
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
├── models/                          our trained models (step 5)                    ~820 MB
│   ├── p4/m2_big.txt                  LightGBM matcher (the "main judge")
│   ├── p4b/ce.pt                      fine-tuned e5-small cross-encoder, 1 epoch (#4-#6; the "second opinion")
│   ├── p11/ce_s2P.pt                  the same cross-encoder trained 2 epochs (#7)
│   ├── p13/ckpt/ce_bge_pool.pt        bge-reranker-v2-m3 cross-encoder, fine-tuned on AWS (#8c / #9, the default)
│   ├── p13/r/r_bge_pool_*             its cached scores (test India / US / France band, holdout) -- the byte check
│   ├── p19/r/r_bge_pool_test_France  bge scores of France's idf144 band (#9ab)
│   ├── p17/stack_sup-bge-l21.0_US.txt house-number stacker, US only (#8c)
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
# pretrained base weights, downloaded once; the pipeline then runs offline (HF_HUB_OFFLINE=1)
# e5-small (MIT, 118M): the e5 cross-encoder of #4-#7, the tests and the holdout's --ce-missing fallback
.venv/bin/python -c "from transformers import AutoModel, AutoTokenizer as T; \
  AutoModel.from_pretrained('intfloat/multilingual-e5-small'); T.from_pretrained('intfloat/multilingual-e5-small')"
# bge-reranker-v2-m3 (Apache-2.0, 568M, 2.3 GB): only to score the bge band fresh (no p13/r cache). On a slow link
# set HF_HUB_DISABLE_XET=1 (xet transfers stalled here), or pull it on an AWS box
.venv/bin/python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; \
  M.from_pretrained('BAAI/bge-reranker-v2-m3'); T.from_pretrained('BAAI/bge-reranker-v2-m3')"
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

Path C still needs `p13/ckpt/ce_bge_pool.pt` from here: the bge cross-encoder was fine-tuned on AWS
(`probes/P13-aws-rerank`, 76 min on an A10G) and `run.py` stops with instructions when it is missing.

Download `amc26-offload/models/` and place the files exactly here:

| Drive file | Put it at | Size | sha256 |
|---|---|---|---|
| `models/p4/m2_big.txt` | `data/work/p4/m2_big.txt` | 67 MB | `598e51c859cf85bf7d59f75135959fcc47e4c71e05ebe6ae3a2414695e485f03` |
| `models/p4b/ce.pt` | `data/work/p4b/ce.pt` | 42 MB | `16013613bbef1c9548a7d91b73978e0be6f833a6007ab45f2460644e1b55c789` |
| `models/p10/pruner.txt` | `data/work/p10/pruner.txt` | 1 MB | `02ab27da2fca67220fc8377eb881c858aa4871c0eaeacd9efa75bed4eb885a52` |
| `models/p11/ce_s2P.pt` (#7) | `data/work/p11/ce_s2P.pt` | 43 MB | `4dc0d8826a9efb3a3d1d956130b5c9af139af6941b77596fc1a5719196c3b9af` |
| `models/p13/ckpt/ce_bge_pool.pt` (#8c; also on S3 `amc26-p13/results/p13-rerank-0925-2257/ckpt/`) | `data/work/p13/ckpt/ce_bge_pool.pt` | 624 MB | `36a946769b50322fda057f3693c7e5577e5f476f28ae95e2c4343fa267b72de7` |
| `work/p13/r/` (#8c; its cached scores) | `data/work/p13/r/r_bge_pool_test_{India,US,France}.parquet`, `r_bge_poolck_holdout.parquet` | 39 MB | — |
| `work/p19/r/` (#9ab; bge scores of the 63,393 France band pairs new on the idf144 candidates) | `data/work/p19/r/r_bge_pool_test_France.parquet` | 1.3 MB | — |
| `models/p17/stack_sup-bge-l21.0_US.txt` (#8c) | `data/work/p17/stack_sup-bge-l21.0_US.txt` | 0.7 MB | `46116715d9821b8a143d05c1ba6c35c40ada7a93dc1ae9523c7fc6cceb15d184` |
| `models/p31/c_hn.txt` (#31 France matcher, P26's AWS job; also on S3 `amc26-p26/results/p26-full-hn-0926-1533/`) | `data/work/p31/c_hn.txt` | 50 MB | `7b48aaede1c4d9ff0a2acf60f09650b307953b64b4c224e3559aee3f020cf36c` |
| `work/p26/bge/r/`, `work/p29/r/`, `work/p31/r/` (#31; bge scores of the France band pairs new on c_hn / the rebuild) | `data/work/{p26/bge,p29,p31}/r/r_bge_pool_test_*.parquet` | 1.5 MB | — |
| `work/p30/r/r_pool_France.parquet` (`sub31_veto` only; bge scores of the confident France pairs) | `data/work/p30/r/r_pool_France.parquet` | 15 MB | — |

**bge needs a GPU to score fresh.** With P13's cached scores above, #8c / #9 need no GPU at all. Scoring the bge band
anew needs two things:
- the HF base `BAAI/bge-reranker-v2-m3` (2.3 GB, downloaded once; >1 GB pulls go through AWS);
- a GPU: ~20 min on an A10G, ~2 h on the laptop's RTX 3050 for the ~1.24M pruned band pairs.

```bash
rclone copy gdrive:amc26-offload/models data/work/     # or copy the files by hand into those paths
sha256sum data/work/p4/m2_big.txt data/work/p4b/ce.pt data/work/p10/pruner.txt data/work/p11/ce_s2P.pt \
    data/work/p13/ckpt/ce_bge_pool.pt data/work/p17/stack_sup-bge-l21.0_US.txt data/work/p31/c_hn.txt   # must match
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
| `p9/fr_idf144/`, `p9/fr31/` (#9ab / #31) | France's worlds: the idf144 vectorizer, its reblocked candidates, X and m2 p; the `(n)`-fixed `rf_test_sx_France` and c_hn's p | ~55 min |

**Version rule — important.** The caches were built by commit `cef4e2d` (25 Sep 22:43) or later. Feature caches are
matched **by file name, not by code version**: if your code cleans text differently from the code that built them,
the pipeline silently reuses stale features. Check out the same commit Harsh used when he uploaded the caches
(`git checkout <commit>`), or delete the affected files so they rebuild. Score caches are safe: their file names
contain the model's id, so a new model always re-scores.

### 6b. Training-side caches — for experimenting with retraining

`amc26-offload/work/` also holds the **train-side** files (~3.1 GB, same `rclone copy` as above brings them):
`train_source*.parquet`, `train_ground_truth.parquet`, `gt_pairs.parquet`, `s1_folds*` (the 10 folds: 0 = holdout,
1 = early stopping, 2–9 = training), `rf_train_*` (record features), `p4/cands_train_full_*` (train candidate
lists), `p4/X/train_*` (the 65 signals for every train candidate pair) and `p4b/trainband_*` (cross-encoder
training pairs). With them a retrain skips ~1 h 45 min of candidate search + features.

**Do the old model weights help when experimenting?**

| You change | Keep the old weights? |
|---|---|
| **LightGBM settings** (leaves, learning rate, folds, sampling) | **No** — train a new model from `p4/X/train_*` (~45 min) and compare it with the old one on the holdout. Save it under a new name; the old `m2_big.txt` stays the baseline to beat. LightGBM does not fine-tune from old weights in any useful way here |
| **Features** (add / change a signal) | **No** — the old model expects exactly the old 65 columns. Rebuild the X files (delete `p4/X/*`), retrain |
| **Cross-encoder** (more data, epochs, another model) | **Partly** — you can continue training from `p4b/ce.pt`, but a clean run from the base e5-small (~25 min GPU) is the fair comparison |
| **Pruner** (threshold τ or its features) | τ only: **yes**, no retrain (`model.TAU`). New features: retrain (74 s) |
| **Only thresholds / France cut-off** | **Yes** — no training at all; `--t-country France=<t>` and the cached scores make it seconds |

Always compare on the **full-corpus holdout** (`run.py --split holdout --stage model --ce-missing fallback`, #5 = 0.9888,
#6 = 0.9885, #7 = 0.9888, #8c = 0.9899)
and log it (`ledger.py`). A change is worth keeping at ≥ +0.0005.

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
| pruner + US stacker (the stacker is refit from the holdout) | — | — | ~2 min + one holdout run |
| bge cross-encoder | — (AWS-trained, downloaded) | — (downloaded) | — (downloaded: `run.py` does not retrain it) |
| score test pairs (pruner + LightGBM) | cached | ~30 min | ~30 min |
| cross-encoder rescoring | cached (`p13/r`) | cached (`p13/r`, from `models/`) | cached, or ~2 h on the RTX 3050 / ~20 min on an A10G |
| decide + write + validate | ~4 min | ~4 min | ~4 min |
| **Total** | **a few minutes** | **~2–2.5 h** | **~5–6 h** |

## 9. Other useful commands

| What | Command |
|---|---|
| Score our own holdout (the number we trust; #8c = 0.9899) | `.venv/bin/python code/business_entity_resolution/src/run.py --split holdout --stage model --ce-missing fallback` (without the flag it bge-scores ~570k competitor pairs: GPU + HF weights) |
| A France option instead of #8c | add `--recipe sub9a` / `sub9b` / `sub9s` / `sub9d` |
| An earlier submission | add `--recipe sub7` / `sub6` / `sub5` / `sub4` |
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

- The command now produces **submission #31** (the default recipe `sub31`, public LB **0.9856**, our best):
  - #8c's India / US: the pruner, m2_big, P13's bge cross-encoder (t 0.71) and P17's US stacker;
  - France: `france.py`'s chain (README section 2), t 0.9868: #9ab's IDF-fixed world rebuilt with the `(n)` house
    numbers and the R1 name fillers, P26's c_hn matcher, the bge blend, then P29's key tier, P28's same-address rule
    (bxneg) and P29's cap.
  - `--t-country France=0.9818` gives `sub31_t9818`; `--recipe sub31_veto` (+ P30's bge veto) gives `sub31_veto`;
    `--recipe sub31_small` (same matching, 4.42 candidates per S1) gives `sub31_small`.
  - Holdout (India / US, = #8c): 0.98990 (half 1 0.98956). France has no labels.
- Needs, beyond #9ab's files: `data/work/p31/c_hn.txt` and the France r top-ups `p26/bge/r`, `p29/r`, `p31/r`
  (`p30/r/r_pool_France.parquet` for `sub31_veto`); see step 5.
- Cold check: France rebuilt from the raw data under the current code (P21's learned région map included) gives
  `sub31`'s France lines exactly: 0 of 259,452 differ (`probes/P9-integrate/cold31.py`, 20 min). With the model and
  score files of step 5 present, a cold run needs ~3 h on the laptop; with step 6's caches (`p9/fr_idf144/`,
  `p9/fr31/` included) it takes ~5 min.
- `--recipe sub9ab`, `sub8c`, `sub9a`, `sub9b`, `sub9s`, `sub7`, `sub6`, `sub5`, `sub4` were each verified byte for byte
  against their staged files. `sub9d` predates P21's learned région map, which moves 1,039 of its France rows by
  design: delete `data/work/p9/fr_idfdom/` to rebuild it under the map.
- Best leaderboard so far: **0.9856** (#31, 26 Sep).
