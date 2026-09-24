# amc-26

Amazon ML Challenge 2026 — **business entity resolution**, macro F0.5. `docs/PROBLEM.md` holds the
rules, output format, metric, data findings and open questions; read it before touching the pipeline.

## Environment

- **Use `.venv/bin/python`, never `python3`.** System Python is 3.14 with a broken pandas install.
- pandas 3 gives string columns `StringDtype`, not `object` — dtype checks must use
  `pd.api.types.is_string_dtype`, or text columns silently vanish.
- Data is in `data/student_resource/dataset/{train,test}/*.tsv`. Read with `sep="\t",
  quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False`. 15 GB RAM — one source at a time.

## Commands

| | |
|---|---|
| validate a submission (from `data/student_resource/`) | `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` |
| framework tests | `.venv/bin/python test_sm.py` |
| rehearse a job in the real container | `.venv/bin/python sm_launch.py --data data/train --test-data data/test --local` |
| account readiness | `.venv/bin/python sm_launch.py --check --instance ml.g4dn.xlarge` |
| metric self-check | `.venv/bin/python code/business_entity_resolution/src/metric.py` |
| ledgers | `.venv/bin/python ledger.py show exp` · `show pipe` · `selftest` |

## Rules

- Never commit `data/` (Amazon's dataset, plus notes that quote it) or any `*.tsv` / `output/`.
- **Disqualifiers:** any model must be MIT/Apache-2.0 and ≤ 8B params, run locally. No hosted
  LLM/embedding APIs (Bedrock, OpenAI, Claude), no geocoding, no registries, no internet data.
- Every test S1 row — France included — must be in the output. Never filter on `{US, India}`.
- The `sagemaker` SDK **is** installed and works (v3 is a namespace package, so
  `sagemaker.__version__` does not exist — import `sagemaker.core` / `sagemaker.train`).
  `sm_launch.py` submits via boto3 and asks the SDK only for the DLC image URI.
- New accounts have **0 SageMaker training quota** — `--check` first. `--local` works without it.
- Never read hyperparameters from argv on SageMaker — the toolkit's `str()` rebuild is lossy.
- During the 72 hours: smallest change that moves the leaderboard. No refactors, no abstractions.
  Every experiment keeps its holdout F0.5 in the commit message so the approach doc writes itself.
