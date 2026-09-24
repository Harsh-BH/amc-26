# Amazon ML Challenge 2026

72-hour team hackathon on an Amazon catalog dataset. Live leaderboard during the run,
private leaderboard on the full test set afterwards.

## Dates (IST)

| | |
|---|---|
| Registration closes | **20 Sep 2026, 23:59** |
| Hackathon | **25 Sep 00:00 (midnight) → ~end of 27 Sep** — 72 h; problem + dataset drop at the start. Exact end time unconfirmed |
| Top 50 announced | 2 Oct 2026 |
| Grand Finale (virtual) | 7 Oct 2026, 10:00–15:00 — top 10 present to Amazon Scientists |

Team of **2–4** (press coverage says 3–4; the Unstop form allows 2). Grad year 2027 or 2028,
full-time engineering, one team per student, cross-college allowed.

Deliverables: a zip of code/notebook **plus a 1–2 page approach document**. The document is read
for finale selection — a good leaderboard score with a thin writeup does not reach the top 10.

## Setup

```bash
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install pandas numpy scikit-learn lightgbm pillow pyarrow
.venv/bin/python metrics.py     # self-check
```

System Python here is 3.14 with a broken half-installed pandas — use the venv, not `python3`.
torch 2.14.0+cu130 is installed and CUDA works on the local RTX 3050 (6 GB).

## Hour 0 checklist

```bash
.venv/bin/python baseline.py --train data/train.csv --sample 20000   # fast signal check
.venv/bin/python baseline.py --train data/train.csv --target <col> --test data/test.csv
.venv/bin/python submit.py --doc approach.pdf --preds submission.csv # preflight before upload
```

1. Read the problem PDF and **confirm the metric** — `metrics.py` assumes SMAPE (2025's metric).
2. Run the baseline on a subsample to get a number on the board within the first hour.
3. Only then start modelling. Keep the median-guess SMAPE as your floor.

## Files

| | |
|---|---|
| `metrics.py` | SMAPE, matching Amazon's stated formula. `python metrics.py` runs the checks. |
| `baseline.py` | TF-IDF (word + char) → LightGBM, log-space target. Sniffs the schema. |
| `submit.py` | Packs the code zip and pre-flights the doc page count, NaNs, row count. |
| `sm_train.py` | SageMaker entry point. You supply `solution.py`; it owns the SM filesystem contract. |
| `sm_launch.py` | Submit / wait / fetch SageMaker jobs, or `--local` in the real container. Spot on by default. |
| `sm_requirements.txt` | Installed in the container at job start — only what the DLC lacks. |
| `test_sm.py` | 29 offline tests for the framework. `python test_sm.py` |
| `docs/EMBEDDINGS.md` | Which embedding model to reach for, by problem shape. Bedrock verified for ap-south-1. |
| `docs/SAGEMAKER.md` | Day-1 order of operations, instance pricing, and the credit traps. |

## Compute plan

$200 AWS credits ($100 signup + $100 as you explore), +$100 for the top 500 at the 48h mark.
Credits cover SageMaker 250h `ml.t3.medium` notebook + 50h training, and S3 5 GB.

`ml.t3.medium` is **CPU-only** (2 vCPU, 4 GB) — it cannot fine-tune a vision-language model, and
$200 is roughly 30–50 GPU-hours total. The 2024 top teams trained on Kaggle's free GPUs.

Plan: AWS for the dataset, S3, and orchestration; **Kaggle/Colab for the training loop**; the
local RTX 3050 (6 GB) for fast iteration on small models only. Decide this before the 25th.

## Prior problem statements

- **2023** — predict product length from tabular catalog features
- **2024** — extract entity values (weight, volume, dimensions) from ~256k product images
- **2025** — "Smart Product Pricing": multimodal text + image → price regression, scored on SMAPE

Another multimodal catalog task is the safe bet, which is what `baseline.py` is shaped for.
