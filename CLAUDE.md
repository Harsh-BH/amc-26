# amc-26

Amazon ML Challenge 2026 — see README.md for dates, rules and the compute plan.

## Environment

- **Use `.venv/bin/python`, never `python3`.** System Python is 3.14 with a broken pandas install.
- pandas 3 gives string columns `StringDtype`, not `object` — dtype checks must use
  `pd.api.types.is_string_dtype`, or text columns silently vanish.

## Commands

| | |
|---|---|
| metric self-check | `.venv/bin/python metrics.py` |
| baseline self-check | `.venv/bin/python baseline.py --demo` |
| preflight submission | `.venv/bin/python submit.py --doc approach.pdf --preds submission.csv` |

## Rules

- Never commit `data/` — it is Amazon's dataset and it is gitignored. Same for `submission.csv`.
- `metrics.py` assumes SMAPE from 2025. **Confirm against the 2026 problem PDF before trusting any score.**
- During the 72 hours: smallest change that moves the leaderboard. No refactors, no abstractions.
  Every experiment keeps its holdout SMAPE in the commit message so the approach doc writes itself.
