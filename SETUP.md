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
| `work/p30/r/r_pool_France.parquet` (the veto of `sub31_veto`, `sub34_veto`, `sub37*`; bge scores of the confident France pairs) | `data/work/p30/r/r_pool_France.parquet` | 15 MB | — |
| `models/p34/a10_h0.txt` (#34 / #37's a10 stacker; also on S3 `amc26-offload/final/models/p34/`) | `data/work/p34/a10_h0.txt` | 0.7 MB | `2ecb55bb0c626bc724c85b832884f4dda274d5499dde661255ad029285285052` |
| `models/p55/fr_plain_s{0,1,2}.txt` (#clean's P55 France combiner, 3 seeds; S3 `amc26-offload/final/models/p55/`) | `data/work/p55/` | 0.7 MB each | `4341e582…` / `efa788a6…` / `95b418bf…` (first 16 hex) |
| `work/p48/r_lo_France.parquet` (#clean: P48's bge scores of the France pairs outside the CE band, AWS GPU) | `data/work/p48/r_lo_France.parquet` | 2.3 MB | `ea4142ad…` |
| `work/p36/morph_test.json` (the 44 brand syllables; rebuilt from the test names when absent) | `data/work/p36/morph_test.json` | 0.3 kB | `dbfbd6ef…` |
| `work/p38/r/r_qwen_pool_test_France.parquet` (`sub_iu54_fr38*` / `best`: Qwen3-Reranker-4B + LoRA scores of France's band, AWS L40S; S3 `amc26-p38/results/p38-pool-0927-0145/r/`) | `data/work/p38/r/` | 7.2 MB | — |
| `models/p50/fullzoo_s{0..4}.txt` (the final India / US combiner, 5 seeds; `full_s*` = without the zoo columns) | `data/work/p50/` | 0.7 MB each | S3 `amc26-offload/final/models/p50/` |
| `work/p10/q_test_{India,US}.parquet` (P10's pruner score of every blocked test pair: a combiner column) | `data/work/p10/` | 170 / 136 MB | — |
| `work/p45/oof/r_all_test_{India,US}.parquet` (P45: the bge CE on every #8c test argmax pair, AWS GPU) | `data/work/p45/oof/` | 71 / 56 MB | — |
| `work/p50/members_test_{India,US}.parquet` (the 2 house-number GBDTs' and 16 zoo members' test predictions) | `data/work/p50/` | 310 / 238 MB | — |

**Disk-tight machines:** a model file moved to cold storage can leave `<file>.tag` (its sha1[:8], e.g.
`echo 3bd0b930 > data/work/p13/ckpt/ce_bge_pool.pt.tag`); runs whose scores are all cached then resolve the caches
without the file (`model.tag`). Anything that must score with the model still needs the file.

**bge needs a GPU to score fresh.** With P13's cached scores above, #8c / #9 need no GPU at all. Scoring the bge band
anew needs two things:
- the HF base `BAAI/bge-reranker-v2-m3` (2.3 GB, downloaded once; >1 GB pulls go through AWS);
- a GPU: ~20 min on an A10G, ~2 h on the laptop's RTX 3050 for the ~1.24M pruned band pairs.

```bash
rclone copy gdrive:amc26-offload/models data/work/     # or copy the files by hand into those paths
sha256sum data/work/p4/m2_big.txt data/work/p4b/ce.pt data/work/p10/pruner.txt data/work/p11/ce_s2P.pt \
    data/work/p13/ckpt/ce_bge_pool.pt data/work/p17/stack_sup-bge-l21.0_US.txt data/work/p31/c_hn.txt \
    data/work/p34/a10_h0.txt   # must match
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
# the final (sub77, LB 0.988292), byte for byte: the base, then the unseen country's change sets (21 s)
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --recipe sub66_final \
    --drop-sibling-cands --out output/sub66_final_small   # sub66_final = LB 0.987391, 4.42 candidates per S1
.venv/bin/python code/business_entity_resolution/src/final_build.py output/sub66_final_small output/sub76
#   md5 matching a7b2cfe0…, candidates c8850dee…; --upto 41 -> output/sub77 (+ P41 brand adds, md5 9e4fb898…)
.venv/bin/python code/business_entity_resolution/src/package.py --team Yeti --out output/sub76 --dest output/final
```

`final_build.py` reads its change sets from `src/final_changes.parquet` (inside the zip) or from
`data/work/p78/final_changes.parquet` (S3 / Drive `amc26-offload/final/`). They are 22,562 rows: P67q, P70a, P69 t975ce,
N54's removals, P71a, P41. `probes/P78-final-pkg2/extract.py` rebuilds them from the probes' staged folders. Each staged
folder came from its probe's own command:
- `output/sub67q`: P67, `build67.py qgate sub66_final sub67 sub67q` in `probes/P67-fr-synth/` (the synthetic-French CE
  column + Qwen gate; the CE and Qwen scores come from GPU runs, see its RESULT.md).
- `output/sub70a`: P70, `probes/P70-fr-roles/p70.py`.
- `output/sub66_t975ce`: P69, `p69.py 0.975 …/t975` → `build69ce.py 0.99 …/t975ce` → `splice.py …/t975ce output/sub66_final
  output/sub66_t975ce` in `probes/P69-fr-loosen/`.
- `output/sub71a`: P71, `probes/P71-fr-noiseword/build71.py a output/sub71a` (label-free word list from its `work/lens_France.csv`).
- `output/sub41_on66`: `probes/P70-fr-roles/p41_on.py`.
- N54: `probes/P74-n54-adapt/build74.py` on #75a, which is `france.decoy_mask`, the same as run.py's `DECOY` hook.

It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (1,732,544 rows each — one per test Source 1
entity, France included), checks the format, then runs the official validator. Expect `PASS`. A later France winner
is one flag away (`--hn-extra 0.9`, `--rescue 0.7`, `--gc 0.6`, `--veto 0.05`, `--t-country France=0.99`; 0 turns a
rule off), on either recipe.

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
| a10 stacker features (India + US; cached in `p9/a10_*` after the first run) | ~6 min | ~6 min | ~6 min + the holdout fit (`run.ensure_a10`) |
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

- **The final: `sub90u` (27 Sep 23:40): `sub88` + 5,429 more France adds from the same repo's pass 3 (same P88 filter) + P80's US owner re-decision (a rival-style stage C: cross-fit M2 + cross-fit CE + competition features, re-deciding only the non-argmax / rival pairs; US fold 0 +0.000287 ± 0.000033; 2,706 US S1 changed); rebuilt byte for byte by `final_patch.py output/sub77 output/sub90u`; public LB 0.989382 [FINAL LB].** Before it, `sub88` (27 Sep 22:32): `sub84` + the India/US combiner upgrade (P70 `ta_ship.py`: 7 generator traces + 5 verified-merge anchors, holdout +0.00005) + 10,689 France adds from the post-hoc ADD rules of the public MIT repo PundarikakshNTripathi/present-day-ber-amlc-2026 (P81 ran its passes 1+2 on our file; P88 dropped veto re-adds and different- / typo-street pairs); rebuilt byte for byte by `final_patch.py output/sub77 output/sub88` (LB 0.989023); before that, `sub84` (27 Sep 20:02): `sub77` + the evening France change sets — xlf (address-less, legal form differs only), groupe swaps, P73 #79q, P76 R3, P80 (address-less, legal form kept), broad type-word veto, sub68fr, never-candidate noise swaps, narrow P73 veto; rebuilt byte for byte by `final_build.py … --upto 41` then `final_patch.py output/sub77 output/sub84`; public LB 0.988475. Previous final (27 Sep 17:30, P78): `sub77`, public LB 0.988292 = `sub66_final` + the unseen country's
  change sets (P67q +2,793, P70a +424, P69 t975ce +4,904, N54 −2,120, P71a +4,523; India / US unchanged). Rebuild it
  byte for byte with `final_build.py` on `--recipe sub66_final --drop-sibling-cands` (step 7). The zip
  `output/final/Yeti_submission.zip` holds it (md5 520204eb…, validator `--check-ids` PASS on the unzipped zip,
  SageMaker p78-val-zip76), and the previous zip is in `output/final/archive/`. Backups: S3 / Drive `amc26-offload/final/`.
- **The command now produces the final (27 Sep 14:45): `sub66_final`, public LB 0.987391** = #54's India / US (the
  combiner of `iu_stack.py`: P44 + P45's r_all + 16 zoo members, refit on the full holdout by P50; P54's duplicate
  copies) + #37_gc70's France chain with the Qwen band blend (P38), P55's veto, P62's acronym block, P54's duplicate
  step, P64's legal-form veto and P66's address-less exact names. It rebuilds `output/sub66_final`'s matching file
  **byte for byte**; `--drop-sibling-cands` gives the zip's candidate file (7,660,700 pairs, 4.42 per S1).
  `--recipe sub_iu54_frgc70` is the earlier final (#37_gc70's France).
- 27 Sep alternatives: `--recipe sub_clean` (France = `france.chain`, P42 / P41 / P48 / P54 / P47 / P55), `sub_clean_iu34`
  (the same with #34's India / US: 0 differing lines as sets vs its folder), `sub54` (the uploaded #54, LB 0.986685).
  `--iu a10` returns India / US to #34 (byte-identical to #37_gc70 then).
- Needs, beyond #37's files: the combiner inputs and boosters and, for `sub_clean`, P55's boosters and P48's `r_lo` (step 5).
- Leaderboard: 0.986304 (#37_gc70) → 0.986685 (#54) → **0.987391 (final, `sub66_final`)**.

### Before 27 Sep 07:30 (history)

- The command now produces **submission #37_gc70** (the default recipe `sub37`, public LB **0.986304**, our best as of
  27 Sep 01:00), byte for byte (both files; P43, 27 Sep):
  - India / US = #34: #8c (the pruner, m2_big, P13's bge cross-encoder, P17's US stacker) + P34's a10 stacker
    (`a10.py`), t 0.71.
  - The unseen country (`run.unseen()` = test S1 labels absent from train S1 = France): `france.py`'s chain (README
    section 2), t 0.9868: #9ab's IDF-fixed world rebuilt with the `(n)` house numbers and the R1 name fillers, P26's
    c_hn matcher, the bge blend, then P29's key tier, P28's same-address rule (bxneg), P30's veto and P29's cap; after
    the decision the empty-S1 rescue (p ≥ 0.90) and P37's group completion (p ≥ 0.70).
  - **`--recipe sub37_small`** = the final zip's files (`output/sub37_final_small`): the same matching file,
    7,658,102 candidate pairs = 4.42 per S1; official validator PASS with `--check-ids`.
  - `--recipe sub37_hn90` (= `--hn-extra 0.9`) and `--rescue 0.7` rebuild P37's staged `sub37_gc70_hn90` /
    `sub37_gc70_r70`.
  - Holdout (India / US report half): 0.98984 with a10 (#8c 0.98956). France has no labels.
- Needs, beyond #31's files: `data/work/p34/a10_h0.txt` and `p30/r/r_pool_France.parquet`; see step 5.
- **No country is named in a decision.** Every unseen-country branch keys on `run.unseen()`; the organisers allow a
  stricter rule for "countries unseen in training" defined that way (27 Sep ruling).
- `--recipe sub31*`, `sub9ab`, `sub8c`, `sub9a`, `sub9b`, `sub9s`, `sub7`, `sub6`, `sub5`, `sub4` were each verified byte
  for byte against their staged files before the UNSEEN refactor (P9). `sub9d` predates P21's learned région map,
  which moves 1,039 of its France rows by design: delete `data/work/p9/fr_idfdom/` to rebuild it under the map.
- Cold check (P9, #31): France rebuilt from the raw data gives `sub31`'s France lines exactly (0 of 259,452 differ,
  `probes/P9-integrate/cold31.py`, 20 min). With the model and score files of step 5, a cold run needs ~3 h on the
  laptop; with step 6's caches (`p9/fr_idf144/`, `p9/fr31/` included) ~12 min (6 min once `p9/a10_*` exists).
- Leaderboard: 0.9856 (#31) → 0.985647 (#34_veto) → 0.98596 (#34v_r90) → **0.986304 (#37_gc70)**; 27 Sep:
  see the 27 Sep status above.
