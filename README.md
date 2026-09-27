# Amazon ML Challenge 2026

72-hour team hackathon. The problem is **business entity resolution**: match noisy business records
across 3 sources, scored on macro F0.5. **Read [`docs/PROBLEM.md`](docs/PROBLEM.md) first** — rules,
output format, metric, data findings, traps and open questions in one place.

## Dates (IST)

| | |
|---|---|
| Hackathon | **25 Sep 00:00:50 → 27 Sep 23:59:49** — max **5 submissions/day** |
| Top 100 announced | then submit methodology details |
| Top 50 results | 2 Oct 2026 — PPIs for Applied Scientist Intern |
| Grand Finale (virtual) | 7 Oct 2026, 10:00–15:00 — top 10 present to Amazon Scientists |

Team of **2–4**. Grad year 2027 or 2028, full-time engineering, one team per student,
cross-college allowed. One Unstop login per person at a time — simultaneous logins can end the run.

Deliverables: `matching_results.tsv` uploads during the run, then `<team_name>_submission.zip`
(outputs + runnable code + filled `Documentation_template.md`, no page limit). Layout in `docs/PROBLEM.md`.

**[FINAL LB] 0.988292** — `sub76` (27 Sep 16:56), in `output/final/Yeti_submission.zip`. It rebuilds byte for byte with
`run.py --recipe sub66_final --drop-sibling-cands` then `final_build.py` (SETUP.md §7).

## Setup

```bash
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install pandas numpy scikit-learn lightgbm pillow pyarrow
```

System Python here is 3.14 with a broken half-installed pandas — use the venv, not `python3`.
torch 2.14.0+cu130 is installed and CUDA works on the local RTX 3050 (6 GB). RAM is 15 GB and
the full train split is ~1.3 GB of TSV — load one source at a time.

Data: unzip `student_resource.zip` into `data/` → `data/student_resource/{dataset,utils,README.md}`.

## Hour 0 checklist

1. Read `docs/PROBLEM.md`. Send the open questions through the Google Form.
2. Score everything on the test-like `holdout` world with `src/metric.py` (macro F0.5).
3. Get a simple baseline on the board: block within country on name tokens + address city,
   accept only confident pairs, give each S2/S3 ID to at most one S1.
4. Before every upload:
   ```bash
   cd data/student_resource && python3 utils/validate_submission.py \
       --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```

## Files

| | |
|---|---|
| `docs/PROBLEM.md` | The problem, rules, data findings, traps, open questions. |
| `code/business_entity_resolution/src/` | The pipeline, where the final zip expects it. `metric.py` = macro F0.5, `io_tsv.py` = TSV → parquet. |
| `ledger.py`, `ledger/` | Experiment + pipeline ledgers. `.venv/bin/python ledger.py show exp` / `show pipe`. |
| `probes/` | tmux Opus 5.5 probe runner and each probe's brief + RESULT.md. |
| `sm_train.py` | SageMaker entry point. You supply `solution.py`; it owns the SM filesystem contract. |
| `sm_launch.py` | Submit / wait / fetch SageMaker jobs, or `--local` in the real container. Spot on by default. |
| `sm_requirements.txt` | Installed in the container at job start — only what the DLC lacks. |
| `test_sm.py` | 29 offline tests for the framework. `.venv/bin/python test_sm.py` |
| `docs/EMBEDDINGS.md` | Local embedding models. Its Bedrock section is **banned** for this problem. |
| `docs/SAGEMAKER.md` | Day-1 order of operations, instance pricing, and the credit traps. |

## Compute plan

$200 AWS credits ($100 signup + $100 as you explore), +$100 for the top 500 at the 48h mark.
Credits cover SageMaker 250h `ml.t3.medium` notebook + 50h training, and S3 5 GB.

The problem is text-only and CPU-friendly: string features + a GBDT on blocked candidates.
A GPU helps only for embedding-based blocking or a small cross-encoder — the local RTX 3050
(6 GB) handles MiniLM/BGE-sized models; SageMaker (`sm_launch.py`) for anything bigger.
Any model must be MIT/Apache-2.0 and ≤ 8B params, run locally — no hosted APIs.
