# Submission queue

Upload on the play page: **Matching Results File** = `matching_results.tsv`, **Code File** = the zip.
Max 5 per day; the day-1 slots expire at 23:59 IST on 25 Sep. Each staged folder under `output/` already
passed the official validator. After an upload, tell the master the leaderboard score so it goes in the
pipeline ledger (`ledger.py show pipe`).

"Dense F0.5" = holdout S1 scored while every train S1 competes, the closest proxy for the 1.73M-S1 test set.

| # | Status | Files | Holdout F0.5 | What it is |
|---|---|---|---|---|
| 1 | staged | `output/matching_results.tsv` + `output/code_s0.zip` | 0.1775 (plain) | exact name + first address segment. Sanity only — **skip it**, #2 tests the same upload path |
| 2 | **staged, validator PASS** — upload this | `output/sub2/matching_results.tsv` + `output/sub2/code.zip` | **0.8215 dense** (precision 0.994) | P2 transliteration → P1 normaliser → exact name/website key within state + address confirmation, unique best S1. Test: 1,540,960 of 1,732,544 S1 matched (France 86%, India 87%, US 92%) |

**Upload first:** the highest-numbered staged row.

To regenerate #2: `.venv/bin/python code/business_entity_resolution/src/rules.py test` (13 min, per country), then the validator.
