# Setup — run the pipeline on your own machine

For teammates (team **Yeti**). One command rebuilds both submission files from the raw data. Start with
[APPROACH.md](APPROACH.md) if you want to know what the pipeline does; this page is only how to run it.

## 1. What you need

| | |
|---|---|
| OS | Linux or macOS (tested on Arch Linux) |
| Python | **3.12** |
| RAM | 16 GB (every step fits a 3 GB cap, but leave room) |
| Disk | ~20 GB free (data 5 GB + intermediate files) |
| GPU | NVIDIA with ≥ 4 GB VRAM strongly recommended (the cross-encoder step). Without one it runs on CPU, much slower |
| Access | GitHub repo `Harsh-BH/amc-26` (private — ask Harsh to add you) and the Drive folder `amc26-offload` (private — ask Harsh to share it with your email) |

## 2. Get the code

```bash
git clone https://github.com/Harsh-BH/amc-26.git
cd amc-26
```

## 3. Install (~5–10 min)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
# the one pretrained model (MIT, 118M) — downloaded once; the pipeline then runs offline
.venv/bin/python -c "from transformers import AutoModel, AutoTokenizer as T; \
  AutoModel.from_pretrained('intfloat/multilingual-e5-small'); T.from_pretrained('intfloat/multilingual-e5-small')"
```

Always use `.venv/bin/python`, not a system `python3`.

## 4. Get the data

Either download `student_resource.zip` from the Unstop challenge page, or copy it from the shared Drive folder
`amc26-offload/data/student_resource/`. It must end up here:

```
data/student_resource/dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/student_resource/dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
data/student_resource/utils/validate_submission.py
```

`data/` is gitignored — **never commit it** (it is Amazon's dataset).

## 5. (Recommended) Use our trained models — skips ~3 h of training

Download from the Drive folder `amc26-offload/models/` and place them exactly here:

| Drive file | Put it at | Size | sha256 |
|---|---|---|---|
| `models/p4/m2_big.txt` | `data/work/p4/m2_big.txt` | 67 MB | `598e51c859cf85bf7d59f75135959fcc47e4c71e05ebe6ae3a2414695e485f03` |
| `models/p4b/ce.pt` | `data/work/p4b/ce.pt` | 42 MB | `16013613bbef1c9548a7d91b73978e0be6f833a6007ab45f2460644e1b55c789` |
| `models/p10/pruner.txt` | `data/work/p10/pruner.txt` | 1 MB | `02ab27da2fca67220fc8377eb881c858aa4871c0eaeacd9efa75bed4eb885a52` |

```bash
mkdir -p data/work/p4 data/work/p4b data/work/p10
# (copy the three files in), then check them:
sha256sum data/work/p4/m2_big.txt data/work/p4b/ce.pt data/work/p10/pruner.txt
```

If you have rclone set up for the same Drive: `rclone copy gdrive:amc26-offload/models data/work/`.

Skip this step to retrain everything from scratch — the same command does it, it just takes longer.

## 6. Run

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --out output/
```

It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (one row per test Source 1 entity,
1,732,544 rows each), checks the format, then runs the official validator. Expect `PASS`.

Everything it computes is cached under `data/work/`, so a second run only re-scores and re-writes (minutes).

## 7. How long it takes (measured on a 16-core laptop, RTX 3050)

| Stage | With our models (step 5) | From scratch |
|---|---|---|
| text prep + candidate search (test) | ~36 min | ~36 min |
| record + pair features (test) | ~50 min | ~50 min |
| train-side candidates + features | — | ~1 h 45 min |
| train LightGBM matcher | — | ~45 min |
| train cross-encoder (GPU) | — | ~30 min |
| score test pairs (LightGBM) | ~1 h (less with the pruner) | ~1 h |
| cross-encoder rescoring (GPU) | ~18 min | ~18 min |
| decide + write + validate | ~3 min | ~3 min |
| **Total** | **~2.5–3 h** | **~5.5–6 h** |

## 8. Other useful commands

| What | Command |
|---|---|
| Score our own holdout (the number we trust; #5 = 0.9888) | `.venv/bin/python code/business_entity_resolution/src/run.py --split holdout --stage model` |
| Reproduce submission #4 instead | add `--recipe sub4` |
| Pipeline regression test | `.venv/bin/python code/business_entity_resolution/src/test_pipeline.py` |
| Validate any submission (from `data/student_resource/`) | `python3 utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test` |
| Experiment ledger | `.venv/bin/python ledger.py show exp` · `show pipe` |

## 9. Troubleshooting

- **Text columns vanish / regex strips letters** — you are on the wrong pandas or Python. Use the venv (pandas 3).
- **Out of memory** — close other heavy programs; every step is designed for ≤ 3 GB, but the OS needs headroom.
  Linux swap helps.
- **No GPU** — it still runs; the cross-encoder steps are slow on CPU. `rerank.py` takes `/tmp/amc-gpu.lock`, so
  two runs never share the GPU.
- **Hugging Face download fails** — run step 3's download once with internet; after that the pipeline is offline
  (`HF_HUB_OFFLINE=1`).

## 10. Current status (update as it changes)

- The command reproduces **submission #5** byte for byte (holdout 0.9888, leaderboard 0.980017).
- Being added now: the **candidate pruner** (#6: 4.83 candidates per S1 instead of 15.9) and a **France threshold
  setting** (leaderboard shows France at 0.88 beats 0.78). Pull the repo again once they are merged.
- Rules: no external data, no APIs, models must be MIT/Apache ≤ 8B — see `docs/PROBLEM.md`. Do not add any.
